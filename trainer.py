"""Regression training with validation-only checkpoint selection."""

import copy
import json
from pathlib import Path
import time

from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from data_process import MoleculeDataset
from model import GNN
from splitters import DataSplitter
from utils import seed_everything

METRIC_NAMES = (
    "mse",
    "rmse",
    "pearson",
    "pearson_p_value",
    "spearman",
    "spearman_p_value",
    "concordance_index",
)


def regression_metrics(labels, predictions):
    """Compute metrics, recording undefined statistics as NaN."""
    mse = mean_squared_error(labels, predictions)
    if len(labels) < 2 or np.ptp(labels) == 0 or np.ptp(predictions) == 0:
        pearson = pearson_p = spearman = spearman_p = float("nan")
    else:
        pearson, pearson_p = pearsonr(labels, predictions)
        spearman, spearman_p = spearmanr(labels, predictions)
    try:
        ci = concordance_index(labels, predictions)
    except ZeroDivisionError:
        ci = float("nan")
    return dict(
        zip(
            METRIC_NAMES,
            map(
                float, (mse, np.sqrt(mse), pearson, pearson_p, spearman, spearman_p, ci)
            ),
        )
    )


def _json_safe(value):
    """Write undefined metrics as JSON null rather than non-standard NaN."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    # Rename within the same directory so interrupted writes cannot mark a fold complete.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def snapshot_state(model):
    """Detach and clone: cpu() alone aliases parameters when training on CPU."""
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


class Trainer:
    def __init__(self, **config):
        if config["model_type"] != "regression":
            raise ValueError("Only regression is supported.")
        self.config = config
        seed_everything(config.get("seed", 0))
        self.modeldir = Path(config["modeldir"]) / f"fold{config['fold']}"
        splitter = DataSplitter(
            config["model_type"],
            config["split_type"],
            cdr_reg_path=config["original_dataset_path"],
            random_state=config.get("split_seed", 2022),
        )
        (train_labels, self.test_smiles), train_indices, val_indices, test_indices = (
            splitter.split_dataset(
                config["fold"], config["num_folds"], return_train_y=True
            )
        )
        self.split_indices = {
            "train": train_indices,
            "validation": val_indices,
            "test": test_indices,
        }
        partitions = [set(indices) for indices in self.split_indices.values()]
        if any(not part for part in partitions):
            raise ValueError(
                "Training, validation, and test partitions must be nonempty."
            )
        if any(
            partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)
        ):
            raise ValueError("Overlapping train/validation/test row positions.")
        if set.union(*partitions) != set(range(len(splitter.CDR))):
            raise ValueError("Splits must cover every response row exactly once.")
        self.scaler = None
        if config.get("use_scaler", False):
            self.scaler = StandardScaler().fit(train_labels.reshape(-1, 1))

        # Share immutable tables and base graphs across split views. __getitem__
        # clones graphs before attaching row-specific targets and omics tensors.
        self.train_dataset = MoleculeDataset(
            root=Path(config["original_dataset_path"]).parent,
            original_dataset_path=config["original_dataset_path"],
            indices=train_indices,
            scaler=self.scaler,
            config=config,
        )
        self.val_dataset = copy.copy(self.train_dataset)
        self.val_dataset.indices = val_indices
        self.test_dataset = copy.copy(self.train_dataset)
        self.test_dataset.indices = test_indices

        self.device = torch.device(
            f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu"
        )
        loader_options = {
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "pin_memory": self.device.type == "cuda",
        }
        if config["num_workers"] > 0:
            loader_options.update(persistent_workers=True, prefetch_factor=4)
        self.train_loader = DataLoader(
            self.train_dataset, shuffle=True, **loader_options
        )
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **loader_options)
        self.test_loader = DataLoader(
            self.test_dataset, shuffle=False, **loader_options
        )
        example = self.train_dataset[0]
        self.model = GNN(
            num_layer=config["num_layer"],
            raw_data=None,
            cell_out_dim=config["cell_out_dim"],
            node_dim=example.x.shape[1],
            edge_dim=example.edge_attr.shape[1],
            hidden_dim=config["hidden_dim"],
            hidden_dim_2=config["hidden_dim_2"],
            num_layer_2=config["num_layer_2"],
            nhead=None,
            num_tasks=1,
            gnn_type=config["gnn_type"],
            gnn_type_interaction=config["gnn_type_interaction"],
            config=config,
            drop_ratio=config["drop_out"],
            gnn_drop_ratio=config["gnn_drop_out"],
            pos_embedd=None,
        ).to(self.device)
        self.loss_fct = torch.nn.MSELoss()
        print(
            f"Split sizes: { {name: len(indices) for name, indices in self.split_indices.items()} }"
        )
        print(
            f"Using {self.device}; model parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        
        # seed_everything(config.get("seed", 0))

    def test(self, datagenerator, model, epo=0, val=True, return_res=False):
        """Evaluate without changing the caller's train/eval mode.

        ``epo`` and ``val`` remain accepted for compatibility with prior callers.
        """
        was_training = model.training
        labels, predictions = [], []
        model.eval()
        try:
            with torch.no_grad():
                for batch in datagenerator:
                    batch = batch.to(self.device)
                    score = model(batch, epo)
                    labels.extend(batch.y.detach().cpu().reshape(-1).tolist())
                    predictions.extend(score.detach().cpu().reshape(-1).tolist())
        finally:
            model.train(was_training)
        if not labels:
            raise ValueError("Cannot evaluate an empty dataset.")
        if self.scaler is not None:
            labels = (
                self.scaler.inverse_transform(np.asarray(labels).reshape(-1, 1))
                .reshape(-1)
                .tolist()
            )
            predictions = (
                self.scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1))
                .reshape(-1)
                .tolist()
            )
        if return_res:
            return labels, predictions
        metrics = regression_metrics(labels, predictions)
        return labels, predictions, *(metrics[name] for name in METRIC_NAMES)

    def train(self):
        config = self.config
        if config["train_epoch"] < 1:
            raise ValueError("train_epoch must be positive.")
        self.modeldir.mkdir(parents=True, exist_ok=True)
        _write_json(self.modeldir / "config.json", config)
        _write_json(self.modeldir / "split_indices.json", self.split_indices)
        if self.scaler is not None:
            _write_json(
                self.modeldir / "scaler.json",
                {
                    "mean": self.scaler.mean_.tolist(),
                    "scale": self.scaler.scale_.tolist(),
                },
            )
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["lr"], weight_decay=config["decay"]
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=5)
        best_rmse = float("inf")
        best_state, best_epoch = None, None
        patience = 0
        history = []
        start = time.monotonic()
        print("Training started", flush=True)
        for epoch in range(config["train_epoch"]):
            epoch_start = time.monotonic()
            self.model.train()
            loss_sum, sample_count = 0.0, 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                # Omics encoders use BatchNorm, which needs two training examples.
                if batch.y.numel() == 1:
                    continue
                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(batch, epoch).reshape(-1)
                loss = self.loss_fct(predictions, batch.y.float().reshape(-1))
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite training loss at epoch {epoch + 1}."
                    )
                loss.backward()
                optimizer.step()
                sample_count += batch.y.numel()
                loss_sum += loss.item() * batch.y.numel()
            if not sample_count:
                raise RuntimeError("No training batch contained at least two examples.")

            result = self.test(self.val_loader, self.model, epoch)
            metrics = dict(zip(METRIC_NAMES, result[2:]))
            if not np.isfinite(metrics["rmse"]):
                raise RuntimeError(f"Non-finite validation RMSE at epoch {epoch + 1}.")
            if epoch >= config["warm_start"] and config["use_scheduler"]:
                scheduler.step(metrics["mse"])
            record = {
                "epoch": epoch + 1,
                "train_mse": loss_sum / sample_count,
                "validation": metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.monotonic() - epoch_start,
            }
            history.append(record)
            _write_json(self.modeldir / "history.json", history)
            print(
                f"Epoch {epoch + 1}: train MSE={record['train_mse']:.6f}, validation RMSE={metrics['rmse']:.6f}, seconds={record['seconds']:.1f}",
                flush=True,
            )
            if metrics["rmse"] <= best_rmse:
                best_rmse, best_epoch = metrics["rmse"], epoch + 1
                best_state = snapshot_state(self.model)
                patience = 0
            else:
                patience += 1
                if patience > config["early_stop_epoch"]:
                    print(
                        f"Early stopping; selected epoch {best_epoch} by validation RMSE.",
                        flush=True,
                    )
                    break

        self.model.load_state_dict(best_state)
        self.save_model()
        labels, predictions, *values = self.test(
            self.test_loader, self.model, epoch, val=False
        )
        test_metrics = dict(zip(METRIC_NAMES, values))
        pd.DataFrame(
            {
                "row_position": self.split_indices["test"],
                "smiles": self.test_smiles,
                "label": labels,
                "pred": predictions,
            }
        ).to_csv(self.modeldir / "test_pred.csv", index=False)
        valid_table = PrettyTable(["epoch", *METRIC_NAMES])
        for record in history:
            valid_table.add_row(
                [
                    record["epoch"],
                    *[f"{record['validation'][name]:.4f}" for name in METRIC_NAMES],
                ]
            )
        (self.modeldir / "valid_markdowntable.txt").write_text(
            valid_table.get_string() + "\n"
        )
        test_table = PrettyTable(list(METRIC_NAMES))
        test_table.add_row([f"{test_metrics[name]:.4f}" for name in METRIC_NAMES])
        (self.modeldir / "test_markdowntable.txt").write_text(
            test_table.get_string() + "\n"
        )
        # This is the completion marker. It is written only after all artifacts.
        _write_json(
            self.modeldir / "metrics.json",
            {
                "best_epoch": best_epoch,
                "best_validation_rmse": best_rmse,
                "test": test_metrics,
                "seconds": time.monotonic() - start,
            },
        )
        print(
            f"Finished: test RMSE={test_metrics['rmse']:.6f}; artifacts: {self.modeldir}",
            flush=True,
        )

    def predict(self):
        labels, predictions = self.test(self.test_loader, self.model, return_res=True)
        frame = pd.DataFrame(
            {"smiles": self.test_smiles, "label": labels, "pred": predictions}
        )
        self.modeldir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.modeldir / "test_pred.csv", index=False)
        return frame

    def save_model(self):
        self.modeldir.mkdir(parents=True, exist_ok=True)
        torch.save(snapshot_state(self.model), self.modeldir / "model.pt")

    def load_pretrained(self, path):
        """Load a full state dictionary, including legacy DataParallel weights."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        state = state.get("state_dict", state)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state)

    def load_matching_weights(self, checkpoint_path):
        """Explicitly load only matching keys/shapes from a transfer checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {
            key.replace("drug_model_encoder_2d", "drug_model_encoder"): value
            for key, value in state.items()
        }
        current = self.model.state_dict()
        matched = {
            key: value
            for key, value in state.items()
            if key in current and value.shape == current[key].shape
        }
        current.update(matched)
        self.model.load_state_dict(current)
        print(
            f"Loaded {len(matched)} / {len(current)} state entries from {checkpoint_path}."
        )
