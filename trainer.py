import os
import numpy as np
import pandas as pd
import json

# from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    f1_score,
    log_loss,
)
import copy
import time
import pickle
import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

import argparse
from cmath import inf

from data_process import MoleculeDataset
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter

from model import GNN
import warnings
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from utils import seed_everything
from splitters import DataSplitter
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")



def compute_label_distribution(dataset):
    try:
        y = dataset.data.y
    except:
        y = torch.tensor(np.array(dataset))
    if y.ndim > 1 and y.shape[1] == 1:  # If y is shape [N, 1], flatten it
        y = y.view(-1)
    unique_labels, counts = torch.unique(y, return_counts=True)
    total = counts.sum().item()
    for label, count in zip(unique_labels, counts):
        percent = (count.item() / total) * 100
        print(f"Label {label.item()}: {percent:.2f}% ({count.item()} samples)")


def load_dataset(model_type, split_type, dataset, fold, num_folds, cdr_reg_path):

    splitter = DataSplitter(model_type, split_type, cdr_reg_path=cdr_reg_path)

    raw_train_data, train_indices, val_indices, test_indices = splitter.split_dataset(
        fold, num_folds, True
    )

    print(
        "Num train: {}, Num val {}, Num test {}".format(
            len(train_indices), len(val_indices), len(test_indices)
        )
    )

    return (
        raw_train_data,
        train_indices,
        val_indices,
        test_indices,
    )


class Trainer:
    def __init__(self, **config):
        # model_drug = transformer()
        self.config = config

        self.modeldir = config["modeldir"] + "/fold" + str(config["fold"])
        
        (raw_train_data, test_smiles), train_indices, val_indices, test_indices = (
            load_dataset(
                config["model_type"],
                config["split_type"],
                None,
                config["fold"],
                config["num_folds"],
                cdr_reg_path=config["original_dataset_path"],
            )
        )
        
        self.test_smiles = test_smiles

        # Fit scaler only on training data
        self.scaler = None
        if config["model_type"] == "regression" and config["use_scaler"]:

            self.scaler = StandardScaler()
            self.scaler.fit(raw_train_data.reshape(-1, 1))

        # self.dataset.set_scaler(self.scaler)
        # print("Set scaler ...")

        # print(self.dataset[train_indices].y[:10])

        if self.config.get("train_all"):
            print("Train model with all data samples ...")
            train_indices = train_indices + val_indices + test_indices
            self.dataset = MoleculeDataset(
                os.path.join("data", config["model_type"]),
                indices=train_indices,
                config=config,
                original_dataset_path=config["original_dataset_path"],
            )
            self.train_loader = DataLoader(
                self.dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
            )
            self.val_loader = None
            self.test_loader = None

        elif self.config.get("train_val"):
            print("Train model with all train + val samples ...")

            self.train_dataset = MoleculeDataset(
                os.path.join("data", config["model_type"]),
                indices=train_indices + val_indices,
                config=config,
                original_dataset_path=config["original_dataset_path"],
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
            )
            self.val_loader = None

            self.val_dataset = MoleculeDataset(
                os.path.join("data", config["model_type"]),
                indices=test_indices,
                config=config,
                original_dataset_path=config["original_dataset_path"],
            )
            self.test_loader = DataLoader(
                self.val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
            )

        else:
            print("Train model with train samples ...")
            self.train_dataset = MoleculeDataset(
                os.path.join("data", config["model_type"]),
                indices=train_indices,
                config=config,
                original_dataset_path=config["original_dataset_path"],
            )
            self.val_dataset = MoleculeDataset(
                os.path.join("data", config["model_type"]),
                indices=val_indices,
                config=config,
                original_dataset_path=config["original_dataset_path"],
            )
            self.test_dataset = MoleculeDataset(
                os.path.join("data", config["model_type"]),
                indices=test_indices,
                config=config,
                original_dataset_path=config["original_dataset_path"],
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                pin_memory=True, persistent_workers=True, prefetch_factor=4
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
                pin_memory=True, persistent_workers=True, prefetch_factor=4
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"],
                pin_memory=True, persistent_workers=True, prefetch_factor=4
            )
            
            print("Train dataset size: ", len(self.train_dataset))
            print("Val dataset size: ", len(self.val_dataset))
            print("Test dataset size: ", len(self.test_dataset))

        print("Load dataset ...")
        self.model = GNN(
            num_layer=config["num_layer"],
            raw_data=None,
            cell_out_dim=config["cell_out_dim"],
            node_dim=self.train_dataset[0].x.shape[1],
            edge_dim=self.train_dataset[0].edge_attr.shape[1],
            hidden_dim=config["hidden_dim"],
            hidden_dim_2=config["hidden_dim_2"],
            num_layer_2=config["num_layer_2"],
            nhead=config.get("nhead", None),
            num_tasks=1,
            gnn_type=config["gnn_type"],
            gnn_type_interaction=config["gnn_type_interaction"],
            config=config,
            drop_ratio=config["drop_out"],
            gnn_drop_ratio=config["gnn_drop_out"],
            pos_embedd=config.get("pos_embedd", None),
            gnn_drop_edge_ratio=config["gnn_edge_drop_out"]
        )
        print("Load model ...")

        print(
            "Number of model parameters: ",
            sum(p.numel() for p in self.model.parameters()),
        )
        # print("Number of GNN parameters: ", sum(p.numel() for p in self.model.gnns.parameters()))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}")
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")

        if self.config["model_type"] == "regression":
            self.loss_fct = torch.nn.MSELoss()

        elif self.config["model_type"] == "classification":
            self.loss_fct = torch.nn.BCEWithLogitsLoss()

        self.freeze_pre = False

        if "checkpoint_path" in config and config["checkpoint_path"] != "":
            if self.config["freeze_pre"]:
                self.freeze_pre = True
                print("Freezing drug encoder for 5 epoch ...")

        seed_everything(0)

    def test(self, datagenerator, model, epo, val=True, return_res=False):
        y_label = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for i, v in enumerate(
                tqdm(
                    datagenerator,
                    desc=(
                        f"Validating epoch {epo}"
                        if val
                        else f"Testing for best checkpoint"
                    ),
                    disable=True,
                )
            ):
                # score,label = model(v)
                v = v.to(self.device)
                label = v.y
                score = self.model(v, epo)
                if self.config["model_type"] == "regression":
                    logits = torch.squeeze(score).detach().cpu().numpy()

                if self.config["model_type"] == "classification":
                    # label = label.long()
                    m = torch.nn.Sigmoid()
                    logits = torch.squeeze(m(score)).detach().cpu().numpy()

                label_ids = label.to("cpu").numpy()
                y_label = y_label + label_ids.flatten().tolist()
                y_pred = y_pred + logits.flatten().tolist()

        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
        model.train()

        if not return_res:

            if self.config["model_type"] == "regression":
                if self.scaler:
                    y_label = self.scaler.inverse_transform(
                        np.array(y_label).reshape(-1, 1)
                    ).reshape(-1)
                    y_pred = self.scaler.inverse_transform(
                        np.array(y_pred).reshape(-1, 1)
                    ).reshape(-1)

                return (
                    y_label,
                    y_pred,
                    mean_squared_error(y_label, y_pred),
                    np.sqrt(mean_squared_error(y_label, y_pred)),
                    pearsonr(y_label, y_pred)[0],
                    pearsonr(y_label, y_pred)[1],
                    spearmanr(y_label, y_pred)[0],
                    spearmanr(y_label, y_pred)[1],
                    concordance_index(y_label, y_pred),
                )
            if self.config["model_type"] == "classification":

                return (
                    roc_auc_score(y_label, y_pred),
                    average_precision_score(y_label, y_pred),
                    f1_score(y_label, outputs),
                    log_loss(y_label, outputs, labels=[0, 1]),
                    y_pred,
                    y_label,
                )
        else:
            if self.config["model_type"] == "regression":
                if self.scaler:
                    y_label = self.scaler.inverse_transform(
                        np.array(y_label).reshape(-1, 1)
                    ).reshape(-1)
                    y_pred = self.scaler.inverse_transform(
                        np.array(y_pred).reshape(-1, 1)
                    ).reshape(-1)
            return y_label, y_pred

    def train(self):
        lr = self.config["lr"]
        decay = self.config["decay"]
        train_epoch = self.config["train_epoch"]
        self.model = self.model.to(self.device)
        # model_max = copy.deepcopy(self.model)
        best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=decay,
        )

        scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.7, patience=5)

        valid_metric_record = []
        test_metric_record = []

        if self.config["model_type"] == "regression":
            min_val_MSE = 10000

            valid_metric_header = [
                "# epoch",
                "MSE",
                "RMSE",
                "Pearson Correlation",
                "with p-value",
                "Spearman Correlation",
                "with p-value2",
                "Concordance Index",
            ]
            test_metric_header = [
                "MSE",
                "RMSE",
                "Pearson Correlation",
                "p-value",
                "spearman",
                "s_p-value",
                "Concordance Index",
            ]

        if self.config["model_type"] == "classification":
            max_auc = 0
            valid_metric_header = ["# epoch", "AUROC", "AUPRC", "F1"]
            test_metric_header = ["AUROC", "AUPRC", "F1"]

        table = PrettyTable(valid_metric_header)
        test_table = PrettyTable(test_metric_header)
        float2str = lambda x: "%0.4f" % x

        print("******** Go for Training ********")
        os.makedirs(self.modeldir, exist_ok=True)
        t_start = time.time()
        es = 0

        if self.freeze_pre:
            for param in self.model.drug_model_encoder.parameters():
                param.requires_grad = False

        for epo in range(train_epoch):
            t = time.time()
            self.model.train()
            for i, v in enumerate(
                tqdm(
                    self.train_loader,
                    desc=f"Training epoch {epo}",
                    disable=True,
                    leave=False,
                )
            ):

                v = v.to(self.device)
                label = v.y
                if label.size(0) == 1:
                    continue
                opt.zero_grad()
                score = self.model(v, epo)

                if self.config["model_type"] == "regression":
                    label = label.float()
                    n = torch.squeeze(score, 1).float()
                    loss = self.loss_fct(n, label)

                if self.config["model_type"] == "classification":
                    n = torch.squeeze(score, 1)
                    loss = self.loss_fct(n, label)

                loss.backward()
                opt.step()
        
            
            if self.val_loader is not None and self.test_loader is not None:
                if self.config["model_type"] == "regression":
                    ### regression: MSE, Pearson Correlation, with p-value, Concordance Index
                    y_true, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = (
                        self.test(self.val_loader, self.model, epo, True)
                    )
                    lst = ["epoch " + str(epo)] + list(
                        map(
                            float2str, [mse, rmse, person, p_val, spearman, s_p_val, CI]
                        )
                    )
                    valid_metric_record.append(lst)
                    
                    if epo >= self.config["warm_start"] and self.config["use_scheduler"]:
                        scheduler.step(mse)

                    print(
                        "Validation at Epoch "
                        + str(epo + 1)
                        + ", RMSE: "
                        + str(rmse)[:7]
                        + " , Time: "
                        + str(time.time() - t)[:7]
                    )

                    if rmse <= min_val_MSE:
                        min_val_MSE = rmse
                        es = 0

                        print(f"========> Epoch {epo+1} Val - RMSE= {rmse}")

                        best_state_dict = {
                            k: v.cpu() for k, v in self.model.state_dict().items()
                        }

                    else:
                        es += 1
                        # print("Counter {} of {}".format(es, self.config['early_stop_epoch']))
                        if es > self.config["early_stop_epoch"]:
                            print(
                                "Early stopping with best_mse: ",
                                str(min_val_MSE)[:7],
                                "and mse for this epoch: ",
                                str(mse)[:7],
                                "...",
                            )
                            break

                if self.config["model_type"] == "classification":
                    auc, auprc, f1, loss, logits, _ = self.test(
                        self.val_loader, self.model, epo, True
                    )
                    lst = ["epoch " + str(epo)] + list(map(float2str, [auc, auprc, f1]))
                    valid_metric_record.append(lst)
                    
                    if epo >= self.config["warm_start"] and self.config["use_scheduler"]:
                        scheduler.step(auc)

                    print(
                        "Validation at Epoch "
                        + str(epo + 1)
                        + ", AUROC: "
                        + str(auc)[:7]
                        + " , AUPRC: "
                        + str(auprc)[:7]
                        + " , F1: "
                        + str(f1)[:7]
                    )

                    if auc > max_auc:

                        max_auc = auc
                        es = 0

                        best_state_dict = {
                            k: v.cpu() for k, v in self.model.state_dict().items()
                        }

                    else:
                        es += 1
                        # print("Counter {} of {}".format(es, self.config['early_stop_epoch']))
                        if es > self.config["early_stop_epoch"]:
                            print(
                                "Early stopping with best_auc: ",
                                str(max_auc)[:7],
                                "and AUC for this epoch: ",
                                str(auc)[:7],
                                "...",
                            )
                            break
                table.add_row(lst)
                print("-" * 20)
                
            
        if self.val_loader is not None:
            prettytable_file = os.path.join(self.modeldir, "valid_markdowntable.txt")
            with open(prettytable_file, "w") as fp:
                fp.write(table.get_string())

        if self.test_loader is not None:
            if self.val_loader is not None: # have validation to get the checkpoints
                self.model.load_state_dict(best_state_dict)

            if self.config["model_type"] == "regression":
                y_true, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = (
                    self.test(self.test_loader, self.model, epo, True)
                )
                test_metric_record = list(
                    map(float2str, [mse, rmse, person, p_val, spearman, s_p_val, CI])
                )

                print(f"========> Test - RMSE= {rmse}")

            if self.config["model_type"] == "classification":
                auc, auprc, f1, loss, y_pred, _ = self.test(
                    self.test_loader, self.model, epo, True
                )
                test_metric_record = list(map(float2str, [auc, auprc, f1]))
                print(f"========> Test - AUC= {auc}")

            test_table.add_row(test_metric_record)
            prettytable_file = os.path.join(self.modeldir, "test_markdowntable.txt")
            with open(prettytable_file, "w") as fp:
                fp.write(test_table.get_string())

        print("******** Training Finished ********")

        tf = open(os.path.join(self.modeldir, "config.json"), "w")
        json.dump(self.config, tf)
        tf.close()

    def predict(self):
        lr = self.config["lr"]
        decay = self.config["decay"]
        train_epoch = self.config["train_epoch"]
        self.model = self.model.to(self.device)
        # model_max = copy.deepcopy(self.model)
        best_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=decay,
        )

        if self.test_loader is not None:
            y_label, y_pred = self.test(self.test_loader, self.model, 0, True, True)

        df = pd.DataFrame(
            {"smiles": self.test_smiles, "label": y_label, "pred": y_pred}
        )
        df.to_csv(os.path.join(os.path.dirname(self.config['checkpoint_path']), "test_pred.csv"))

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.modeldir, "model.pt"))

    def load_matching_weights(self, checkpoint_path):
        # Load checkpoint (assume it's a state_dict or has one inside)
        print(f"Load model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        state_dict = {
            k.replace("drug_model_encoder_2d", "drug_model_encoder"): v
            for k, v in state_dict.items()
        }

        model_dict = self.model.state_dict()
        matched_state_dict = {}

        # Iterate and match by key and shape
        for key in state_dict:
            if key in model_dict:
                if state_dict[key].shape == model_dict[key].shape:
                    matched_state_dict[key] = state_dict[key]
                else:
                    print(
                        f"Skip loading {key}: not found or shape mismatch "
                        f"(ckpt: {state_dict[key].shape}"
                    )  # , model: {model_dict.get(key, 'missing').shape})")

        # Update model weights
        model_dict.update(matched_state_dict)
        self.model.load_state_dict(model_dict)

        print(
            f"Loaded {len(matched_state_dict)} / {len(model_dict)} layers from checkpoint."
        )

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == "cuda":
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device("cpu"))

        if next(iter(state_dict))[:7] == "module.":
            # the pretrained model is from data-parallel module
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)
