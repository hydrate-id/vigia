# Vigia

Detect online gambling domains by name and page content. Two models, one API.

## What it does

- **Domain model** — Logistic Regression over character n-grams of a domain name.
- **Content model** — Logistic Regression over visible HTML text.
- Runs both on every predict, then fuses the result.
- Learn from newly registered domains nightly.

## Deploy modes

### 1. Docker (full: API + worker scheduler)

Runs FastAPI **and** an APScheduler job at 02:00 that enriches NRD data and
retrains both models. `dataset.csv` and `content.csv` are writable.

```bash
docker build -t vigia .
docker run -p 8000:8000 -v "$PWD/data:/app/data" vigia
```

Worker mode is enabled by `VIGIA_WORKER=1` (set in the Dockerfile). In this
mode `POST /dataset` works and nightly retrain runs in-process.

### 2. Vercel (serverless, predict-only)

Read-only filesystem (except `/tmp`) and no cron, so:

- Predict only. `POST /dataset` returns `403 read-only`.
- Nightly retrain runs on **GitHub Actions**
  (`.github/workflows/nightly-retrain.yml`), which commits the updated
  `data/` models back, then auto-deploys to Vercel.
- Worker (`APScheduler`) is **off** — no `VIGIA_WORKER`.

Deploy (free tier needs a personal account, not an org):

```bash
vercel deploy --prod
```

Env vars: `API_TOKEN` (optional, see below), `SCRAPINGANT_API_KEY` (scrape
fallback when a site blocks the direct request; JS render over datacenter
proxy, budget-capped at ~250 credits per run via `SCRAPE_ANT_BUDGET_CREDITS`).

## Endpoints

- `GET /predict/{domain}?token=...` — fetches the page, scores both models,
  fuses: `p = 0.7 * content + 0.3 * domain`. Returns `is_gambling`,
  `confidence_percent`, `source`, plus both verdicts.
- `POST /dataset` — body `{"domain": "...", "is_gambling": true|false}`.
  Docker only; on Vercel it is read-only.
- `GET /health` — readiness + row counts.

### Rate limit (Vercel, like nawala)

- No `API_TOKEN` env set → public, 3 req/s per IP.
- `API_TOKEN` env set → requests with `?token=<API_TOKEN>` skip the limit;
  requests without a token still get 3 req/s per IP.

```
GET /predict/google.com
GET /predict/google.com?token=<API_TOKEN>
```

## How content is judged

A page is labeled from its text, not from the domain list it came from:
strong gambling terms (id/en/zh/th/ko) → positive; real text without them →
negative; parking / anti-bot shell / ambiguous → skipped. JS shells that
can't be read are caught by a `gambling_shell` gate (real `sm4.js`/`jsjiami`
anti-bot + CJK disguise title). Chinese recruiting funnels into private chat
servers are caught by a multi-signal funnel gate.

## Nightly job

`ml/nightly.py` — fetch yesterday's NRD from smet.cz, confirmed gambling →
`dataset.csv`, scrape ≤2000 new domains → label by text → `content.csv`, then
retrain both models. Runs on GitHub Actions (or the in-Docker APScheduler in
worker mode).

## Layout

- `app/` — FastAPI, onnxruntime predictor, APScheduler (worker), rate limit
- `ml/` — train, seed, enrich, scrape, labeling, nightly, smoke test
- `data/` — CSV corpora + ONNX models (tracked in git)
- `ml/features.py` imports sklearn; runtime paths use `ml/domain.py` instead
  so the serverless bundle stays free of scikit-learn.

## Data attribution

- Negative seed domains: Cisco Umbrella top-1M.
- Gambling seed domains: TrustPositif mirror (alsyundawy).
- Nightly enrichment: **Newly Registered Domains — smet.cz** (CC BY 4.0).
  Redistribute only with attribution to `smet.cz`.
