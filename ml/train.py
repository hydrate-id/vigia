import csv
import json
import os
import random
import tempfile
from datetime import datetime, timezone

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import (
    CONTENT_CSV,
    DATASET_CSV,
    METRICS_CONTENT_JSON,
    METRICS_JSON,
    MIN_SAMPLES_PER_CLASS,
    MODEL_CONTENT_ONNX,
    MODEL_ONNX,
    MODELS_DIR,
    RANDOM_STATE,
)
from ml.features import make_vectorizer, normalize_domain


def read_dataset(csv_path):
    texts, labels = [], []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                domain = normalize_domain(row.get("domain"))
                if domain is None:
                    continue
                raw = str(row.get("is_gambling", "")).strip().lower()
                if raw in ("1", "true", "yes", "y"):
                    labels.append(1)
                    texts.append(domain)
                elif raw in ("0", "false", "no", "n"):
                    labels.append(0)
                    texts.append(domain)
    except FileNotFoundError:
        return [], []
    return texts, labels


def read_content(csv_path):
    texts, labels = [], []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = str(row.get("is_gambling", "")).strip().lower()
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                if raw in ("1", "true", "yes", "y"):
                    labels.append(1)
                    texts.append(text)
                elif raw in ("0", "false", "no", "n"):
                    labels.append(0)
                    texts.append(text)
    except FileNotFoundError:
        return [], []
    return texts, labels


def _balance_train(x, y, rng):
    idx0 = [i for i, v in enumerate(y) if v == 0]
    idx1 = [i for i, v in enumerate(y) if v == 1]
    target = min(len(idx0), len(idx1))
    keep = rng.sample(idx0, target) + rng.sample(idx1, target)
    return [x[i] for i in keep], [y[i] for i in keep]


def _to_onnx(pipeline, dest):
    model = convert_sklearn(
        pipeline,
        initial_types=[("X", StringTensorType([None, 1]))],
        target_opset=18,
    )
    with open(dest, "wb") as f:
        f.write(model.SerializeToString())


def _atomic_write_bytes(dest, data):
    models_dir = dest.parent
    fd, tmp = tempfile.mkstemp(dir=models_dir, suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_json(dest, obj):
    models_dir = dest.parent
    fd, tmp = tempfile.mkstemp(dir=models_dir, suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def train_model(x_all, y_all, onnx_dest, metrics_dest):
    n1 = sum(y_all)
    n0 = len(y_all) - n1
    if min(n0, n1) < MIN_SAMPLES_PER_CLASS:
        return {"trained": False, "reason": "insufficient_data", "n0": n0, "n1": n1}

    y = np.array(y_all)
    x_train, x_eval, y_train, y_eval = train_test_split(
        x_all, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    rng = random.Random(RANDOM_STATE)
    x_train_b, y_train_b = _balance_train(x_train, y_train.tolist(), rng)
    y_train_b = np.array(y_train_b)

    pipeline = Pipeline(
        [
            ("vec", make_vectorizer()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(x_train_b, y_train_b)

    accuracy = float(pipeline.score(x_eval, y_eval))

    fd, tmp_model = tempfile.mkstemp(dir=onnx_dest.parent, suffix=".onnx")
    os.close(fd)
    try:
        _to_onnx(pipeline, tmp_model)
        os.replace(tmp_model, onnx_dest)
    except BaseException:
        if os.path.exists(tmp_model):
            os.unlink(tmp_model)
        raise

    metrics = {
        "trained": True,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "accuracy": round(accuracy, 4),
        "n_train": int(len(x_train_b)),
        "n_eval": int(len(x_eval)),
        "n0": n0,
        "n1": n1,
    }
    _atomic_write_json(metrics_dest, metrics)
    return metrics


def retrain_domain(csv_path=None, models_dir=None):
    csv_path = csv_path or DATASET_CSV
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    texts, labels = read_dataset(csv_path)
    return train_model(texts, labels, models_dir / "model.onnx", models_dir / "metrics.json")


def retrain_content(csv_path=None, models_dir=None):
    csv_path = csv_path or CONTENT_CSV
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    texts, labels = read_content(csv_path)
    return train_model(
        texts, labels, models_dir / "model_content.onnx", models_dir / "metrics_content.json"
    )
