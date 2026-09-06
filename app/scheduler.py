import logging
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import CONTENT_CSV, DATASET_CSV, MODELS_DIR, RETRAIN_HOUR, RETRAIN_MINUTE
from app.dataset import ensure_content_csv, ensure_csv, read_rows
from ml.train import retrain_content, retrain_domain

log = logging.getLogger("vigia.scheduler")

_LOCK = threading.Lock()


def _need_seed(csv_path):
    rows = read_rows(csv_path)
    if not rows:
        return True
    n1 = sum(1 for v in rows.values() if v == 1)
    n0 = len(rows) - n1
    return min(n0, n1) < 1


def _train_all(csv_path, content_path, models_dir):
    metrics = {
        "domain": retrain_domain(csv_path, models_dir),
        "content": retrain_content(content_path, models_dir),
    }
    log.info("retrain done: %s", metrics)
    return metrics


def bootstrap(domain_predictor, content_predictor, csv_path=None, content_path=None, models_dir=None):
    csv_path = csv_path or DATASET_CSV
    content_path = content_path or CONTENT_CSV
    models_dir = models_dir or MODELS_DIR
    ensure_csv(csv_path)
    ensure_content_csv(content_path)
    with _LOCK:
        if domain_predictor.ready() and content_predictor.ready():
            return
        try:
            if _need_seed(csv_path):
                from ml.seed import seed

                log.info("dataset too small, seed started")
                result = seed(csv_path)
                log.info("seed done: %s", result)
            if not domain_predictor.ready():
                log.info("initial domain retrain")
                retrain_domain(csv_path, models_dir)
                domain_predictor.load()
            if not content_predictor.ready():
                from ml.enrich import bootstrap_content

                log.info("bootstrap content (scrape labeled sample)")
                bootstrap_content(csv_path, content_path)
                log.info("initial content retrain")
                retrain_content(content_path, models_dir)
                content_predictor.load()
        except Exception:
            log.exception("bootstrap failed, model unavailable")


def make_scheduler(domain_predictor, content_predictor, csv_path=None, content_path=None, models_dir=None):
    csv_path = csv_path or DATASET_CSV
    content_path = content_path or CONTENT_CSV
    models_dir = models_dir or MODELS_DIR
    sched = BackgroundScheduler()

    def job():
        with _LOCK:
            try:
                from ml.enrich import enrich

                log.info("NRD enrich started")
                result = enrich(csv_path, content_path, domain_predictor, content_predictor)
                log.info("enrich done: %s", result)
            except Exception:
                log.exception("enrich failed, retrying anyway")
            try:
                _train_all(csv_path, content_path, models_dir)
                domain_predictor.load()
                content_predictor.load()
            except Exception:
                log.exception("scheduled retrain failed")

    sched.add_job(job, CronTrigger(hour=RETRAIN_HOUR, minute=RETRAIN_MINUTE), id="retrain", coalesce=True)
    return sched
