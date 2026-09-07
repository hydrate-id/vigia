import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import (
    CONTENT_CSV,
    DATASET_CSV,
    METRICS_CONTENT_JSON,
    METRICS_JSON,
    MODEL_CONTENT_ONNX,
    MODEL_ONNX,
    MODELS_DIR,
    PREDICT_CACHE_TTL_HOURS,
    PREDICT_CONTENT_WEIGHT,
    PREDICT_THRESHOLD,
)
from app.dataset import append_domain, read_content_rows, read_rows
from app.predictor import Predictor
from ml.domain import normalize_domain

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vigia")

domain_predictor = Predictor(MODEL_ONNX, METRICS_JSON, threshold=PREDICT_THRESHOLD, normalizer=normalize_domain)
content_predictor = Predictor(MODEL_CONTENT_ONNX, METRICS_CONTENT_JSON, threshold=PREDICT_THRESHOLD)

# Worker mode (Docker): run APScheduler + train on boot.
# Serverless (Vercel): predict-only, scheduling via GitHub Actions.
_WORKER = os.environ.get("VIGIA_WORKER", "") == "1"
_SERVERLESS = os.environ.get("VERCEL", "") == "1"


def _check_rate(request):
    from app.ratelimit import allow, client_ip, unlimited_token

    token = request.query_params.get("token")
    if unlimited_token(token):
        return
    ip = client_ip(request)
    if not allow(ip):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


def _meta(path):
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text())


@asynccontextmanager
async def lifespan(app):
    domain_predictor.load()
    content_predictor.load()
    if _WORKER:
        from app.scheduler import bootstrap, make_scheduler

        scheduler = make_scheduler(
            domain_predictor, content_predictor, DATASET_CSV, CONTENT_CSV, MODELS_DIR
        )
        scheduler.start()
        try:
            if not (domain_predictor.ready() and content_predictor.ready()):
                t = threading.Thread(
                    target=bootstrap,
                    args=(domain_predictor, content_predictor, DATASET_CSV, CONTENT_CSV, MODELS_DIR),
                    daemon=True,
                )
                t.start()
            yield
        finally:
            scheduler.shutdown(wait=False)
    else:
        yield


app = FastAPI(title="Vigia", lifespan=lifespan)

_PREDICT_CACHE_TTL_S = PREDICT_CACHE_TTL_HOURS * 3600
_predict_cache = {}
_predict_cache_lock = threading.Lock()


def _model_version():
    dm = (_meta(METRICS_JSON) or {}).get("trained_at")
    cm = (_meta(METRICS_CONTENT_JSON) or {}).get("trained_at")
    return (dm, cm)


def _cache_get(key, version):
    with _predict_cache_lock:
        item = _predict_cache.get(key)
    if (
        item is not None
        and item[0] == version
        and time.time() - item[1] < _PREDICT_CACHE_TTL_S
    ):
        return item[2]
    return None


def _cache_put(key, version, value):
    with _predict_cache_lock:
        if len(_predict_cache) > 100_000:
            _predict_cache.clear()  # ponytail: blunt sweep, version+TTL guard staleness
        _predict_cache[key] = (version, time.time(), value)


class DatasetItem(BaseModel):
    domain: str
    is_gambling: bool


@app.get("/health")
def health():
    rows = read_rows(DATASET_CSV)
    n1 = sum(1 for v in rows.values() if v == 1)
    crows = read_content_rows(CONTENT_CSV)
    cn1 = sum(1 for v in crows.values() if v[0] == 1)
    return {
        "domain_model_ready": domain_predictor.ready(),
        "content_model_ready": content_predictor.ready(),
        "total_rows": len(rows),
        "n_gambling": n1,
        "n_normal": len(rows) - n1,
        "content_total_rows": len(crows),
        "content_n_gambling": cn1,
        "content_n_normal": len(crows) - cn1,
        "domain_model": _meta(METRICS_JSON),
        "content_model": _meta(METRICS_CONTENT_JSON),
    }


def _content_probe(domain):
    from ml.labeling import classify_page
    from ml.scrape import fetch_page

    if not content_predictor.ready():
        return {"status": "model_not_ready", "verdict": None}
    page = fetch_page(domain)
    if page is None:
        return {"status": "fetch_failed", "verdict": None}
    label, status = classify_page(page)
    if label is None:
        return {"status": status, "verdict": None}
    if status == "gambling_shell":
        return {
            "status": status,
            "verdict": {
                "is_gambling": True,
                "confidence_percent": 97.0,
                "probability_gambling": 0.97,
                "model_version": "shell_gate",
            },
        }
    combo = " ".join(
        x for x in (page.get("title", ""), page.get("meta", ""), page.get("text", "")) if x
    ).strip()
    verdict = content_predictor.score_text(combo)
    cmeta = _meta(METRICS_CONTENT_JSON)
    if verdict:
        if status in ("gambling", "gambling_funnel") and verdict["probability_gambling"] < 0.97:
            verdict["is_gambling"] = True
            verdict["confidence_percent"] = 97.0
            verdict["probability_gambling"] = 0.97
        verdict["model_version"] = (cmeta or {}).get("trained_at")
    return {"status": status, "verdict": verdict}


def _fuse(domain_verdict, content_probe):
    p_domain = domain_verdict["probability_gambling"]
    p = p_domain
    source = "domain"
    if content_probe["verdict"]:
        p_content = content_probe["verdict"]["probability_gambling"]
        p = PREDICT_CONTENT_WEIGHT * p_content + (1 - PREDICT_CONTENT_WEIGHT) * p_domain
        source = "domain+content"
    elif content_probe["status"] in ("shell", "parked", "ambiguous"):
        source = f"domain (content={content_probe['status']})"
    elif content_probe["status"] == "fetch_failed":
        source = "domain (content=fetch_failed)"
    elif content_probe["status"] == "model_not_ready":
        source = "domain (content=model_not_ready)"
    is_gambling = p >= PREDICT_THRESHOLD
    confidence = max(p, 1 - p) * 100
    return {
        "domain": domain_verdict["domain"],
        "is_gambling": is_gambling,
        "confidence_percent": round(confidence, 1),
        "probability_gambling": round(p, 4),
        "source": source,
        "domain_verdict": domain_verdict,
        "content_status": content_probe["status"],
        "content_verdict": content_probe["verdict"],
    }


@app.get("/predict/{domain}")
def predict(domain: str, request: Request):
    if _SERVERLESS:
        _check_rate(request)
    if not domain_predictor.ready():
        raise HTTPException(status_code=503, detail="model not ready, retrain in progress")
    normalized = normalize_domain(domain)
    if normalized is None:
        raise HTTPException(status_code=422, detail="invalid domain")
    version = _model_version()
    cached = _cache_get(normalized, version)
    if cached is not None:
        resp = JSONResponse(cached)
        resp.headers["Cache-Control"] = f"s-maxage={int(_PREDICT_CACHE_TTL_S)}"
        return resp
    domain_verdict = domain_predictor.predict(normalized)
    if domain_verdict is None:
        raise HTTPException(status_code=422, detail="invalid domain")
    domain_verdict["model_version"] = version[0]
    content_probe = _content_probe(domain_verdict["domain"])
    result = _fuse(domain_verdict, content_probe)
    _cache_put(normalized, version, result)
    resp = JSONResponse(result)
    resp.headers["Cache-Control"] = f"s-maxage={int(_PREDICT_CACHE_TTL_S)}"
    return resp


@app.post("/dataset")
def add_to_dataset(item: DatasetItem, request: Request):
    if _SERVERLESS:
        raise HTTPException(
            status_code=403,
            detail="read-only: dataset updates run in Docker or via GitHub Actions",
        )
    domain, status, total = append_domain(DATASET_CSV, item.domain, item.is_gambling)
    if domain is None:
        raise HTTPException(status_code=422, detail="invalid domain")
    return {"domain": domain, "status": status, "total_rows": total}
