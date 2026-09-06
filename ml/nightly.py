"""Nightly job for GitHub Actions: enrich data then retrain both models.

Mirrors the old in-app APScheduler cron. Runs fully offline except for the
network calls enrich/seed already make.
"""
import logging
import sys

from app.config import CONTENT_CSV, DATASET_CSV, MODELS_DIR
from app.dataset import ensure_content_csv, ensure_csv, read_rows
from ml.train import retrain_content, retrain_domain

log = logging.getLogger("vigia.nightly")


def _need_seed(csv_path):
    rows = read_rows(csv_path)
    if not rows:
        return True
    n1 = sum(1 for v in rows.values() if v == 1)
    n0 = len(rows) - n1
    return min(n0, n1) < 1


def main():
    logging.basicConfig(level=logging.INFO)
    ensure_csv(DATASET_CSV)
    ensure_content_csv(CONTENT_CSV)

    if _need_seed(DATASET_CSV):
        from ml.seed import seed

        log.info("dataset too small, seed started")
        print("seed:", seed(DATASET_CSV))

    from ml.enrich import bootstrap_content, enrich

    log.info("NRD enrich started")
    print("enrich:", enrich(DATASET_CSV, CONTENT_CSV))

    if not read_rows(DATASET_CSV):
        log.warning("dataset still empty after enrich, nothing to train")
        return 1

    log.info("bootstrap content (scrape labeled sample)")
    print("content bootstrap:", bootstrap_content(DATASET_CSV, CONTENT_CSV))

    metrics = {
        "domain": retrain_domain(DATASET_CSV, MODELS_DIR),
        "content": retrain_content(CONTENT_CSV, MODELS_DIR),
    }
    log.info("retrain done: %s", metrics)
    print("metrics:", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
