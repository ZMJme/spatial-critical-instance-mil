"""Run inside MIL_compare.ipynb after executing its imports and model classes.

Notebook commands:
    %run -i retrain_baselines_patience10_on_CAMELYON16.py --models DSMIL
    %run -i retrain_baselines_patience10_on_CAMELYON16.py --models CLAM_SB
    %run -i retrain_baselines_patience10_on_CAMELYON16.py --models Ours
    %run -i retrain_baselines_patience10_on_CAMELYON16.py --dataset BRACS --models DSMIL
    %run -i retrain_baselines_patience10_on_CAMELYON16.py --models ABMIL GTP MaxPool MeanPool TransMIL

Omit --models to run all seven in the above order. This script trains only;
it does not evaluate the test set or change an existing result in the paper.
"""

import argparse
import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm


ORDER = ["DSMIL", "CLAM_SB", "ABMIL", "GTP", "MaxPool", "MeanPool", "TransMIL"]
MODEL_CHOICES = ORDER + ["Ours", "RNN", "SlideGraph"]
CLASS_NAMES = {
    "Ours": "MyModel",
    "RNN": "RNN",
    "SlideGraph": "SlideGraph",
    "DSMIL": "DSMIL_Model",
    "CLAM_SB": "CLAM_SB_Model",
    "ABMIL": "ABMIL",
    "GTP": "GTP",
    "MaxPool": "MaxPooling",
    "MeanPool": "MeanPooling",
    "TransMIL": "TransMIL",
}


def shared_split(data_path, save_root, expected_train=216, expected_val=54, dataset="CAMELYON16"):
    """Create once, then reload the identical filename lists for every run."""
    split_path = save_root / f"patience10_split_on_{dataset}.json"
    legacy_path = save_root / "patience10_split.json"
    source_path = split_path if split_path.exists() else legacy_path
    # Reuse an existing split verbatim rather than drawing a new partition.
    if source_path.exists():
        with source_path.open(encoding="utf-8") as handle:
            split = json.load(handle)
        if Path(split["data_path"]).resolve() != data_path.resolve():
            raise ValueError(f"Saved split belongs to another data directory: {split_path}")
    else:
        # Preserve the source workflow's os.listdir order and random_state=42.
        files = np.array([
            name for name in os.listdir(data_path)
            if name.endswith(".pth") and (data_path / name).is_file()
        ])
        train_files, val_files = train_test_split(
            files, test_size=0.2, random_state=42
        )
        split = {
            "data_path": str(data_path.resolve()),
            "random_state": 42,
            "test_size": 0.2,
            "train": train_files.tolist(),
            "val": val_files.tolist(),
        }

    train_files = split["train"]
    val_files = split["val"]
    assert len(train_files) == expected_train, f"Expected {expected_train} train slides, got {len(train_files)}"
    assert len(val_files) == expected_val, f"Expected {expected_val} validation slides, got {len(val_files)}"
    assert len(set(train_files)) == len(train_files)
    assert len(set(val_files)) == len(val_files)
    assert set(train_files).isdisjoint(val_files)
    saved_files = set(train_files + val_files)
    current_files = {
        name for name in os.listdir(data_path)
        if name.endswith(".pth") and (data_path / name).is_file()
    }
    if saved_files != current_files:
        raise ValueError("Dataset file list differs from the shared split; check it before training.")
    if not split_path.exists():
        with split_path.open("x", encoding="utf-8") as handle:
            json.dump(split, handle, ensure_ascii=False, indent=2)
    return split, split_path


def train_one(name, data_path, save_root, split, device, class_num=2, dataset="CAMELYON16"):
    # Existing model folder / new model-specific patience10 subfolder.
    save_path = save_root / name / f"{name}_patience10"
    save_path.mkdir(parents=True, exist_ok=True)
    reserved = ["model.pth", "log.log", f"split_on_{dataset}.json", "config.json"]
    for filename in reserved:
        path = save_path / filename
        if path.exists():
            raise FileExistsError(f"Existing run will not be overwritten: {path}")

    model_class = globals()[CLASS_NAMES[name]]
    if name == "Ours":
        model_gnn = GNNModel(384, 256, 256, num_layers=1)
        model_iclassifier = FCLayer(256, class_num)
        model_bclassifier = BClassifier(256, class_num)
        model_mil = MILNet(model_iclassifier, model_bclassifier)
        model = model_class(model_gnn, model_mil).to(device)
    elif name == "CLAM_SB":
        model = model_class(384, class_num, dropout=0.25, k_sample=8).to(device)
    else:
        model = model_class(384, class_num).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_names = {
        "Ours": "0.5 bag CE + 0.5 max-instance CE",
        "DSMIL": "0.5 bag BCEWithLogits + 0.5 max-instance BCEWithLogits",
        "CLAM_SB": "0.7 bag CE + 0.3 instance loss",
    }
    config = {
        "model": name,
        "model_class": CLASS_NAMES[name],
        "dataset": dataset,
        "class_num": class_num,
        "feature_dim": 384,
        "learning_rate": 3e-4,
        "optimizer": "Adam",
        "max_epochs": 100,
        "patience": 10,
        "early_stopping_logic": "Reset to 10 on improvement, then decrement every epoch",
        "consecutive_non_improvements_to_stop": 9,
        "training_loss": loss_names.get(name, "bag CE"),
        "checkpoint_selection": "minimum validation bag CE",
        "train_order": "fixed shared split order; no epoch reshuffling",
        "training_seed": None,
        "data_path": str(data_path.resolve()),
    }
    if name == "CLAM_SB":
        config.update(dropout=0.25, k_sample=8)
    elif name == "Ours":
        config.update(
            graph_encoder="GNNModel(384,256,256,num_layers=1)",
            graph_edges="original spatial edges from get_data; not self-loop-only",
            aggregation_head=f"BClassifier(256,{class_num})",
        )

    for filename, value in [("config.json", config), (f"split_on_{dataset}.json", split)]:
        with (save_path / filename).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)

    logger = logging.getLogger(f"patience10.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for old_handler in logger.handlers[:]:
        logger.removeHandler(old_handler)
        old_handler.close()
    handler = logging.FileHandler(save_path / "log.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.info(json.dumps(config, ensure_ascii=False))

    loss_min = np.inf
    patience = 10
    e_count = patience
    best_epoch = None

    try:
        for epoch in range(100):
            model.train()
            train_losses = []
            for filename in tqdm(split["train"], desc=f"{name} Epoch {epoch} Train"):
                features, edges, label = get_data(str(data_path / filename), device)
                target = label.long().reshape(-1)
                optimizer.zero_grad()

                if name == "DSMIL":
                    instance_logits, bag_logits, _, _ = model(features)
                    one_hot = F.one_hot(target, num_classes=class_num).float()
                    max_logits = instance_logits.max(dim=0, keepdim=True).values
                    loss = (
                        0.5 * F.binary_cross_entropy_with_logits(bag_logits, one_hot)
                        + 0.5 * F.binary_cross_entropy_with_logits(max_logits, one_hot)
                    )
                elif name == "Ours":
                    instance_logits, bag_logits, _, _ = model(features, edges)
                    max_logits = instance_logits.max(dim=0, keepdim=True).values
                    loss = (
                        0.5 * F.cross_entropy(bag_logits, target)
                        + 0.5 * F.cross_entropy(max_logits, target)
                    )
                elif name == "CLAM_SB":
                    instance_loss, bag_logits, _, _ = model(
                        features, label=target, instance_eval=True
                    )
                    loss = 0.7 * F.cross_entropy(bag_logits, target) + 0.3 * instance_loss
                else:
                    # Original MIL_compare classes all accept (features, edges).
                    _, bag_logits, _, _ = model(features, edges)
                    loss = F.cross_entropy(bag_logits, target)

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"{name}: non-finite training loss in {filename}")
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for filename in tqdm(split["val"], desc=f"{name} Epoch {epoch} Val"):
                    features, edges, label = get_data(str(data_path / filename), device)
                    target = label.long().reshape(-1)
                    if name == "DSMIL":
                        _, bag_logits, _, _ = model(features)
                    elif name == "CLAM_SB":
                        _, bag_logits, _, _ = model(features, instance_eval=False)
                    else:
                        _, bag_logits, _, _ = model(features, edges)
                    loss = F.cross_entropy(bag_logits, target)
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError(f"{name}: non-finite validation loss in {filename}")
                    val_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            saved = ""
            if val_loss < loss_min:
                e_count = patience
                loss_min = val_loss
                best_epoch = epoch
                torch.save(model, save_path / "model.pth")
                saved = "saved"

            # Intentionally match the source notebook's countdown semantics.
            e_count -= 1
            info = (
                f"epoch:{epoch}\tremaining:{e_count}\t"
                f"train:{train_loss}\tval:{val_loss}\t{saved}"
            )
            logger.info(info)
            handler.flush()
            print(info)
            if e_count < 1:
                logger.info("Early stopping")
                print("Early stopping")
                break

        summary = {"model": name, "best_epoch": best_epoch, "best_validation_ce": loss_min}
        with (save_path / "training_summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        logger.info("Training complete: %s", summary)
        print(f"{name} weight: {(save_path / 'model.pth').resolve()}")
    finally:
        logger.removeHandler(handler)
        handler.close()
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["CAMELYON16", "BRACS"], default="CAMELYON16")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=ORDER)
    parser.add_argument(
        "--data-path", default=None
    )
    parser.add_argument("--save-root", default=None)
    args = parser.parse_args()
    required = ["get_data"] + [CLASS_NAMES[name] for name in args.models]
    if "Ours" in args.models:
        required += ["GNNModel", "FCLayer", "BClassifier", "MILNet"]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise NameError(
            f"Run the notebook class/function definitions first: {missing}. "
            "Then use %run -i retrain_baselines_patience10_on_CAMELYON16.py ..."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this training run expects cuda:0.")
    if args.dataset == "BRACS":
        class_num, expected_train, expected_val = 4, 99, 25
        default_data = "/data/database/BRACS/pkl_data/old/train/"
        default_root = "./model/MIL/compare/BRACS/"
    else:
        class_num, expected_train, expected_val = 2, 216, 54
        default_data = "/data/database/CAMELYON16/pkl_data/training/20xbags/"
        default_root = "./model/MIL/compare/Camelyon/"
    data_path = Path(args.data_path or default_data)
    if not data_path.is_dir():
        raise FileNotFoundError(data_path)
    save_root = Path(args.save_root or default_root)
    save_root.mkdir(parents=True, exist_ok=True)
    split, split_path = shared_split(data_path, save_root, expected_train, expected_val, args.dataset)
    print("Shared split:", split_path.resolve())
    print("Train:", len(split["train"]), "Val:", len(split["val"]))
    print("Run order:", args.models)
    device = torch.device("cuda:0")
    for name in args.models:
        train_one(name, data_path, save_root, split, device, class_num, args.dataset)


if __name__ == "__main__":
    main()
