import argparse
import io
import random
import tempfile
import urllib.request
import zipfile

from app.config import GAMBLING_URL, SEED_PER_CLASS, UMBRELLA_URL
from app.dataset import ensure_csv, read_rows, save_rows
from ml.features import normalize_domain


def _download(url):
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def _sample_positives(data, per_class, rng):
    sampled = []
    for line in io.StringIO(data.decode("utf-8", errors="replace")):
        domain = normalize_domain(line.strip())
        if domain is None:
            continue
        if len(sampled) < per_class:
            sampled.append(domain)
        else:
            j = rng.randrange(len(sampled))
            sampled[j] = domain
    return sampled


def _sample_negatives(zip_bytes, exclude, per_class):
    neg = []
    seen = set(exclude)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            for raw in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                domain = raw.strip().split(",")[-1]
                domain = normalize_domain(domain)
                if domain is None or domain in seen:
                    continue
                seen.add(domain)
                neg.append(domain)
                if len(neg) == per_class:
                    break
    return neg


def seed(csv_path, pos_url=None, neg_url=None, per_class=SEED_PER_CLASS, seed_value=42):
    pos_url = pos_url or GAMBLING_URL
    neg_url = neg_url or UMBRELLA_URL
    ensure_csv(csv_path)
    existing = read_rows(csv_path)
    rng = random.Random(seed_value)

    print(f"[seed] download positif: {pos_url}")
    pos_bytes = _download(pos_url)
    pos = _sample_positives(pos_bytes, per_class, rng)
    print(f"[seed] positif baru: {len(pos)}")

    exclude = set(existing)
    exclude.update(pos)
    print(f"[seed] download negatif: {neg_url}")
    neg_bytes = _download(neg_url)
    target_neg = len(pos)
    neg = _sample_negatives(neg_bytes, exclude, target_neg)
    print(f"[seed] negatif baru: {len(neg)}")

    for d in pos:
        existing.setdefault(d, 1)
    for d in neg:
        existing.setdefault(d, 0)
    save_rows(csv_path, existing)

    n1 = sum(1 for v in existing.values() if v == 1)
    n0 = len(existing) - n1
    return {"added_pos": len(pos), "added_neg": len(neg), "n0": n0, "n1": n1}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/dataset.csv")
    p.add_argument("--pos-url", default=None)
    p.add_argument("--neg-url", default=None)
    p.add_argument("--per-class", type=int, default=SEED_PER_CLASS)
    args = p.parse_args()
    result = seed(args.csv, args.pos_url, args.neg_url, args.per_class)
    print(result)
