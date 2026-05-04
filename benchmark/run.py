"""
How to use:

    python run.py <config> --dataset <dataset_name>

Arguments:
    config          Path to a YAML model config file (e.g. configs/fasttext.yaml)
    --dataset       Name of the dataset to run (must be defined in the config)

Examples:
    python run.py configs/fasttext.yaml --dataset sst2
    python run.py configs/fasttext.yaml --dataset ag_news

Results are saved as JSON files under benchmark/results/<model_name>/.
"""

import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch

import datasets as _ds
import numpy as np
import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers
from torchTextClassifiers.tokenizers import NGramTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
_ds.disable_progress_bars()

RESULTS_DIR = Path(__file__).parent / "results"


# ── Data Loading ──────────────────────────────────────────────────────────────

def _make_X(texts, titles, cat_arrays):
    """Build the X array: text only (1D) or text + categorical features (2D)."""
    if titles is not None:
        texts = [(t + " " + b) if t is not None else b for t, b in zip(titles, texts)]
    if cat_arrays:
        return np.column_stack([np.array(texts)] + [np.array(c) for c in cat_arrays])
    return np.array(texts)


def _make_y(labels, label_offset, oos_label=None):
    return np.array([oos_label if v is None else int(v) + label_offset for v in labels])


def load_data(dataset_cfg: dict, seed: int):
    text_col   = dataset_cfg["text_col"]
    title_col  = dataset_cfg.get("title_col")
    label_col  = dataset_cfg["label_col"]
    label_offset = dataset_cfg.get("label_offset", 0)
    oos_label    = dataset_cfg.get("oos_label")
    cat_cols_cfg = dataset_cfg.get("categorical_cols", {})
    cat_cols   = list(cat_cols_cfg.keys())
    n_train    = dataset_cfg["train_size"]
    n_val      = dataset_cfg["val_size"]
    hf_config = dataset_cfg.get("hf_config")

    ds = load_dataset(dataset_cfg["hf_path"], hf_config) if hf_config else load_dataset(dataset_cfg["hf_path"])

    # Encoders for string categoricals (vocab size unknown until we see the data)
    cat_encoders = {}  # col -> {str_value: int}

    def _encode_cat(col, vals):
        if col in cat_encoders:
            return np.array([cat_encoders[col].get(v, 0) for v in vals])  # string col: apply mapping, unknown values → 0
        return np.array(vals).astype(int)  # numeric col: cast directly

    def _build_encoders(col_values: dict):
        """Build string→int encoders for cols with no known vocab size (None in config)."""
        for col in cat_cols:
            if cat_cols_cfg.get(col) is None:  # vocab size unknown → infer from training data
                unique_vals = sorted(set(col_values[col]))  # collect unique values from train set
                cat_encoders[col] = {v: i for i, v in enumerate(unique_vals)}  # build string→int mapping
                dataset_cfg["categorical_cols"][col] = len(unique_vals)  # update vocab size in config for ModelConfig

    train_split = dataset_cfg.get("train_split", "train")
    test_split  = dataset_cfg.get("test_split", "test")
    train_data  = ds[train_split].shuffle(seed=seed)

    _build_encoders({col: train_data[col][:n_train] for col in cat_cols})

    def from_split(data, start, end):
        texts  = data[text_col][start:end]
        titles = data[title_col][start:end] if title_col else None
        labels = _make_y(data[label_col][start:end], label_offset, oos_label)
        cats   = [_encode_cat(col, data[col][start:end]) for col in cat_cols]
        return _make_X(texts, titles, cats), labels

    X_train, y_train = from_split(train_data, 0, n_train)
    X_val, y_val = from_split(train_data, n_train, n_train + n_val)

    test_data = ds[test_split].shuffle(seed=seed)
    n_test    = dataset_cfg.get("test_size", len(test_data))
    X_test, y_test = from_split(test_data, 0, n_test)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def build_tokenizer(tok_cfg: dict, X_train: np.ndarray):
    # Text is always column 0 (or the full array if 1D)
    texts = X_train[:, 0].tolist() if X_train.ndim > 1 else X_train.tolist()

    if tok_cfg["type"] == "ngram":
        tokenizer = NGramTokenizer(
            min_count=tok_cfg.get("min_count", 1),
            min_n=tok_cfg.get("min_n", 3),
            max_n=tok_cfg.get("max_n", 6),
            num_tokens=tok_cfg.get("num_tokens", 100000),
            len_word_ngrams=tok_cfg.get("len_word_ngrams", 1),
        )
        tokenizer.train(texts)
    elif tok_cfg["type"] == "wordpiece":
        from torchTextClassifiers.tokenizers import WordPieceTokenizer
        tokenizer = WordPieceTokenizer(
            vocab_size=tok_cfg.get("vocab_size", 10000),
            output_dim=tok_cfg.get("output_dim", 128),
        )
        tokenizer.train(texts)
    else:
        raise ValueError(f"Unknown tokenizer: {tok_cfg['type']}")
    return tokenizer


# ── Save Results ──────────────────────────────────────────────────────────────

def save_results(model_name: str, dataset_name: str, metrics: dict):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_dir = RESULTS_DIR / model_name
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{dataset_name}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved → {json_path}")


# ── Run ───────────────────────────────────────────────────────────────────────

def run(config_path: str, dataset_name: str):
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    seed = cfg.get("seed", 42)
    all_datasets = cfg["datasets"]

    if dataset_name not in all_datasets:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(all_datasets)}")

    dataset_cfg = all_datasets[dataset_name]
    dataset_cfg["name"] = dataset_name

    print(f"\n{'='*50}")
    print(f"Model : {model_name} | Dataset : {dataset_name}")
    print(f"{'='*50}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_cfg, seed)
    num_classes = len(np.unique(y_train))

    tokenizer = build_tokenizer(cfg["tokenizer"], X_train)

    # Categorical variables
    cat_cols_cfg = dataset_cfg.get("categorical_cols", {})
    categorical_vocabulary_sizes = list(cat_cols_cfg.values()) if cat_cols_cfg else None

    m = cfg["model"]
    model_config = ModelConfig(
        embedding_dim=m["embedding_dim"],
        num_classes=num_classes,
        aggregation_method=m.get("aggregation_method", "mean"),
        categorical_vocabulary_sizes=categorical_vocabulary_sizes,
        attention_config=m.get("attention_config"),
        n_heads_label_attention=m.get("n_heads_label_attention"),
    )
    clf = torchTextClassifiers(tokenizer=tokenizer, model_config=model_config)

    t = cfg["training"]
    training_config = TrainingConfig(
        num_epochs=t["num_epochs"],
        batch_size=t["batch_size"],
        lr=t["lr"],
        patience_early_stopping=t.get("patience_early_stopping", 3),
        num_workers=t.get("num_workers", 0),
        raw_labels=False,
        raw_categorical_inputs=False,
        save_path=str(RESULTS_DIR / "models" / model_name / dataset_name),
    )

    t0 = time.time()
    clf.train(X_train, y_train, training_config=training_config, X_val=X_val, y_val=y_val)
    train_time = round(time.time() - t0, 1)

    predict_batch_size = 512 #to avoid predicting every exemple at once --> OOM
    all_preds = []
    for i in range(0, len(X_test), predict_batch_size):
        batch = X_test[i:i + predict_batch_size]
        result = clf.predict(batch, raw_categorical_inputs=False)
        all_preds.append(result["prediction"].squeeze(dim=-1))
    preds = torch.cat(all_preds).numpy()

    metrics = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "tokenizer": cfg["tokenizer"]["type"],
        "embedding_dim": m["embedding_dim"],
        "aggregation_method": m.get("aggregation_method", "mean"),
        "attention_config": m.get("attention_config"),
        "categorical_cols": list(cat_cols_cfg.keys()) if cat_cols_cfg else [],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_accuracy": round(accuracy_score(y_test, preds), 4),
        "test_f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
        "train_time_s": train_time,
        "timestamp": datetime.now().isoformat(),
        "classification_report": classification_report(
            y_test, preds,
            target_names=dataset_cfg.get("class_names"),
            output_dict=True,
        ),
    }

    print(f"Test Accuracy : {metrics['test_accuracy']:.4f}")
    print(f"Test F1 macro : {metrics['test_f1_macro']:.4f}")
    print(f"Train time    : {train_time}s")

    save_results(model_name, dataset_name, metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Chemin vers le fichier YAML du modèle")
    parser.add_argument("--dataset", required=True, help="Nom du dataset à utiliser")
    args = parser.parse_args()
    run(args.config, args.dataset)
