"""Command-line entry point for cross-validated CCMG regression training."""

import argparse
import hashlib
import json
from pathlib import Path

from data_process import DATA_DIR

OMICS = (
    "expr + mutation + methylation",
    "expr + mutation",
    "expr + methylation",
    "mutation + methylation",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="CUDA device index; CPU if CUDA is unavailable",
    )
    parser.add_argument("--gnn_type", default="GIN", choices=["GIN", "GSAGE"])
    parser.add_argument(
        "--split_type",
        default="discovery",
        choices=["discovery", "blind_scaffold", "blind_cell"],
    )
    parser.add_argument("--num_folds", default=10, choices=[5, 10, 20], type=int)
    parser.add_argument(
        "--fold", type=int, help="Run one fold (1-based); by default run all folds"
    )
    parser.add_argument("--drop_out", default=0.5, type=float)
    parser.add_argument("--gnn_drop_out", default=0.1, type=float)
    parser.add_argument("--hidden_dim_2", default=256, type=int)
    parser.add_argument("--omics", default=OMICS[0], choices=OMICS)
    parser.add_argument(
        "--seed", default=0, type=int, help="Model and data-loader random seed"
    )
    parser.add_argument("--split_seed", default=2022, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the four processed GDSC tables",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    return parser


def make_config(args, fold):
    data_dir = args.data_dir.resolve()
    config = {
        "split_type": args.split_type,
        "model_type": "regression",
        "num_layer": 4,
        "num_layer_2": 4,
        "cell_out_dim": 256,
        "hidden_dim": 256,
        "hidden_dim_2": args.hidden_dim_2,
        "gnn_type": args.gnn_type,
        "gnn_type_interaction": args.gnn_type,
        "fold": fold,
        "num_folds": args.num_folds,
        "use_scaler": False,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": 5e-5,
        "decay": 1e-6,
        "train_epoch": args.epochs,
        "early_stop_epoch": 10,
        "drop_out": args.drop_out,
        "gnn_drop_out": args.gnn_drop_out,
        "warm_start": 3,
        "use_scheduler": True,
        "omics": args.omics,
        "gpu": args.gpu,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "wash_smiles": True,
        "brics": False,
        "data_dir": str(data_dir),
        "original_dataset_path": str(data_dir / "CDR_n156813.txt"),
    }
    # Exclude fold so all folds of a configuration share a run directory.
    identity = {key: value for key, value in config.items() if key != "fold"}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    config["modeldir"] = str(
        args.output_dir.resolve() / args.split_type / args.gnn_type / digest
    )
    return config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fold is not None and not 1 <= args.fold <= args.num_folds:
        parser.error("--fold must be between 1 and --num_folds")
    if (
        args.epochs < 1
        or args.batch_size < 2
        or args.num_workers < 0
        or args.hidden_dim_2 < 1
    ):
        parser.error(
            "Require epochs >= 1, batch_size >= 2, num_workers >= 0, hidden_dim_2 >= 1"
        )
    if not 0 <= args.drop_out <= 1 or not 0 <= args.gnn_drop_out <= 1:
        parser.error("Dropout probabilities must be between 0 and 1")

    from trainer import Trainer

    folds = [args.fold] if args.fold is not None else range(1, args.num_folds + 1)
    for fold in folds:
        config = make_config(args, fold)
        fold_dir = Path(config["modeldir"]) / f"fold{fold}"
        complete = fold_dir / "metrics.json"
        if complete.exists():
            saved_config = fold_dir / "config.json"
            if (
                not saved_config.exists()
                or json.loads(saved_config.read_text()) != config
            ):
                raise RuntimeError(
                    f"Existing run configuration does not match: {fold_dir}"
                )
            if not (fold_dir / "model.pt").is_file():
                raise RuntimeError(
                    f"Completed run is missing its checkpoint: {fold_dir}"
                )
            print(f"Skipping completed fold {fold}: {fold_dir}")
            continue
        print(f"Training fold {fold}/{args.num_folds}: {fold_dir}")
        Trainer(**config).train()


if __name__ == "__main__":
    main()
