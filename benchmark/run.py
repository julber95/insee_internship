"""
How to use:
"""

import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

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
    """Construit le tableau X : texte seul (1D) ou texte + catégoriels (2D)."""
    if titles is not None:
        texts = [t + " " + b for t, b in zip(titles, texts)]
    if cat_arrays:
        return np.column_stack([np.array(texts)] + [np.array(c) for c in cat_arrays])
    return np.array(texts)


def _make_y(labels, label_offset):
    return np.array([int(v) + label_offset for v in labels])


def load_data(dataset_cfg: dict, seed: int):
    streaming  = dataset_cfg.get("streaming", False)
    text_col   = dataset_cfg["text_col"]
    title_col  = dataset_cfg.get("title_col")
    label_col  = dataset_cfg["label_col"]
    label_offset = dataset_cfg.get("label_offset", 0)
    cat_cols   = list(dataset_cfg.get("categorical_cols", {}).keys())
    n_train    = dataset_cfg["train_size"]
    n_val      = dataset_cfg["val_size"]

    ds = load_dataset(dataset_cfg["hf_path"], streaming=streaming)

    if streaming:
        n_test = dataset_cfg.get("test_size", 2000)
        total  = n_train + n_val + n_test
        raw_iter = ds[dataset_cfg.get("train_split", "train")].shuffle(seed=seed, buffer_size=10000)
        examples = list(raw_iter.take(total))
        raw = {k: [ex[k] for ex in examples] for k in examples[0]}

        def slice_raw(start, end):
            texts  = raw[text_col][start:end]
            titles = raw[title_col][start:end] if title_col else None
            labels = _make_y(raw[label_col][start:end], label_offset)
            cats   = [[int(raw[col][i]) for i in range(start, end)] for col in cat_cols]
            return _make_X(texts, titles, cats), labels

        X_train, y_train = slice_raw(0, n_train)
        X_val, y_val = slice_raw(n_train, n_train + n_val)
        X_test, y_test = slice_raw(n_train + n_val, total)

    else:
        train_split = dataset_cfg.get("train_split", "train")
        test_split  = dataset_cfg.get("test_split", "test")
        train_data  = ds[train_split].shuffle(seed=seed)

        def from_split(data, start, end):
            texts  = data[text_col][start:end]
            titles = data[title_col][start:end] if title_col else None
            labels = _make_y(data[label_col][start:end], label_offset)
            cats   = [np.array(data[col][start:end]).astype(int) for col in cat_cols]
            return _make_X(texts, titles, cats), labels

        X_train, y_train = from_split(train_data, 0, n_train)
        X_val, y_val = from_split(train_data, n_train, n_train + n_val)

        test_data = ds[test_split].shuffle(seed=seed)
        n_test    = dataset_cfg.get("test_size")
        X_test, y_test = from_split(test_data, 0, n_test if n_test else len(test_data))

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def build_tokenizer(tok_cfg: dict, X_train: np.ndarray):
    # Le texte est toujours la colonne 0 (ou le tableau entier si 1D)
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
        raise ValueError(f"Dataset '{dataset_name}' inconnu. Disponibles : {list(all_datasets)}")

    dataset_cfg = all_datasets[dataset_name]
    dataset_cfg["name"] = dataset_name

    print(f"\n{'='*50}")
    print(f"Model : {model_name} | Dataset : {dataset_name}")
    print(f"{'='*50}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_cfg, seed)
    num_classes = len(np.unique(y_train))

    tokenizer = build_tokenizer(cfg["tokenizer"], X_train)

    # Categorical Variables
    cat_cols_cfg = dataset_cfg.get("categorical_cols", {})
    categorical_vocabulary_sizes = list(cat_cols_cfg.values()) if cat_cols_cfg else None

    m = cfg["model"]
    model_config = ModelConfig(
        embedding_dim=m["embedding_dim"],
        num_classes=num_classes,
        aggregation_method=m.get("aggregation_method", "mean"),
        categorical_vocabulary_sizes=categorical_vocabulary_sizes,
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

    result = clf.predict(X_test, raw_categorical_inputs=False)
    preds = result["prediction"].squeeze().numpy()

    metrics = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "tokenizer": cfg["tokenizer"]["type"],
        "embedding_dim": m["embedding_dim"],
        "aggregation_method": m.get("aggregation_method", "mean"),
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
