import csv
import os
import tempfile
import threading
from pathlib import Path

from ml.domain import normalize_domain

_LOCK = threading.Lock()


def ensure_csv(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("domain,is_gambling\n")


def read_rows(path):
    ensure_csv(path)
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            domain = normalize_domain(row.get("domain"))
            if domain is None:
                continue
            raw = str(row.get("is_gambling", "")).strip().lower()
            if raw in ("1", "true", "yes", "y"):
                rows[domain] = 1
            elif raw in ("0", "false", "no", "n"):
                rows[domain] = 0
    return rows


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".csv")
    os.close(fd)
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["domain", "is_gambling"])
            for domain in sorted(rows):
                writer.writerow([domain, int(rows[domain])])
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def append_domain(path, raw_domain, is_gambling):
    domain = normalize_domain(raw_domain)
    if domain is None:
        return None, None, None
    with _LOCK:
        rows = read_rows(path)
        existed = domain in rows
        rows[domain] = 1 if is_gambling else 0
        save_rows(path, rows)
        return domain, ("updated" if existed else "appended"), len(rows)


def batch_append_domains(path, items):
    with _LOCK:
        rows = read_rows(path)
        added = 0
        for domain, label in items:
            if domain not in rows:
                added += 1
            rows[domain] = 1 if label else 0
        save_rows(path, rows)
        return added, len(rows)


def ensure_content_csv(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("domain,is_gambling,text\n")


def read_content_rows(path):
    ensure_content_csv(path)
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            domain = normalize_domain(row.get("domain"))
            if domain is None:
                continue
            raw = str(row.get("is_gambling", "")).strip().lower()
            label = None
            if raw in ("1", "true", "yes", "y"):
                label = 1
            elif raw in ("0", "false", "no", "n"):
                label = 0
            text = (row.get("text") or "").strip()
            if label is not None and text:
                rows[domain] = (label, text)
    return rows


def save_content_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".csv")
    os.close(fd)
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["domain", "is_gambling", "text"])
            for domain in sorted(rows):
                label, text = rows[domain]
                writer.writerow([domain, int(label), text])
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def append_content_batch(path, items):
    with _LOCK:
        rows = read_content_rows(path)
        added = 0
        for domain, label, text in items:
            if domain not in rows:
                added += 1
            rows[domain] = (1 if label else 0, text)
        save_content_rows(path, rows)
        return added, len(rows)
