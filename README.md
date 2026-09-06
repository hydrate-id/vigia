# Vigia

Detect Indonesian online gambling (judol) domains. Two models, one API.

## What it does

- **Domain model** — Logistic Regression over character n-grams of a domain name.
- **Content model** — Logistic Regression over visible HTML text.
- Runs both on every predict, then fuses the result.
- Retrains itself nightly at 02:00 UTC and learns from newly registered domains.

## Models

Built with scikit-learn, exported to ONNX, served via onnxruntime.

| Model | Input | File |
|---|---|---|
| Domain | domain string | `data/models/model.onnx` |
| Content | visible page text | `data/models/model_content.onnx` |

## Endpoints

- `GET /predict/{domain}` — fetches the page, scores both models, fuses:
  `p = 0.7 * content + 0.3 * domain`. Returns `is_gambling`, `confidence_percent`,
  `source`, plus both `domain_verdict` and `content_verdict` for auditing.
- `POST /dataset` — body `{"domain": "...", "is_gambling": true|false}`. Upserts a
  labeled domain into `dataset.csv`.
- `GET /health` — model readiness + row counts for both corpora.

## How content is judged

A page is labeled from its text, not from the domain list it came from:

- Strong gambling terms (id/en/zh/th/ko) → positive.
- Real text without gambling terms → negative.
- Parking / anti-bot shell / ambiguous → skipped (never labeled).

JS shells that can't be read are caught by a `gambling_shell` gate: real
`<script src="sm4.js">` or `jsjiami` anti-bot, CJK disguise title (`澳门`, `彩民`,
`信息站`, ...) and numeric junk domain. Shell verdict is rule-based (97%).

## Nightly job (02:00 UTC)

1. Pull yesterday's Newly Registered Domains from smet.cz.
2. Domains already on the TrustPositif gambling blocklist → `dataset.csv`.
3. Sample ≤2000 new domains, scrape HTML, label by page text → `content.csv`.
4. Retrain both models, reload.

On first boot with no model: seed a balanced dataset, scrape a labeled content
sample, then train.

## Data attribution

- Negative seed domains: Cisco Umbrella top-1M.
- Gambling seed domains: TrustPositif mirror (alsyundawy).
- Nightly enrichment: **Newly Registered Domains — smet.cz** (CC BY 4.0).
  Redistribute only with attribution to `smet.cz`.
- NRD files close per UTC day, kept 90 days.

## Run

```bash
docker build -t vigia .
docker run -p 8000:8000 -v "$PWD/data:/app/data" vigia
```

Locally:

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
uv run python -m ml.smoke_test
uv run uvicorn app.main:app --port 8000
```

## Layout

- `app/` — FastAPI, onnxruntime predictor, APScheduler 02:00 job
- `ml/` — train, seed, enrich, scrape, labeling, smoke test
- `data/` — CSV corpora + exported models (gitignored)
