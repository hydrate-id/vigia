from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATASET_CSV = DATA_DIR / "dataset.csv"
CONTENT_CSV = DATA_DIR / "content.csv"
MODELS_DIR = DATA_DIR / "models"
MODEL_ONNX = MODELS_DIR / "model.onnx"
METRICS_JSON = MODELS_DIR / "metrics.json"
MODEL_CONTENT_ONNX = MODELS_DIR / "model_content.onnx"
METRICS_CONTENT_JSON = MODELS_DIR / "metrics_content.json"
BLOCKLIST_CACHE = DATA_DIR / "blocklist.txt"

PREDICT_THRESHOLD = 0.5
PREDICT_CONTENT_WEIGHT = 0.7
MIN_SAMPLES_PER_CLASS = 15
SEED_PER_CLASS = 10_000
RANDOM_STATE = 42

RETRAIN_HOUR = 2
RETRAIN_MINUTE = 0

GAMBLING_URL = (
    "https://raw.githubusercontent.com/alsyundawy/TrustPositif/"
    "main/gambling_indonesia_001.txt"
)
UMBRELLA_URL = "https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"

NRD_MANIFEST_URL = "https://smet.cz/nrd/data/manifest.json"
NRD_DAILY_URL = "https://smet.cz/nrd/data/daily/{date}.txt.gz"
NRD_SAMPLE = 2000
NRD_CONF_POS = 0.99
NRD_CONF_NEG = 0.01
BLOCKLIST_TTL_HOURS = 24

SCRAPE_TIMEOUT = 6
SCRAPE_MAX_BYTES = 200_000
SCRAPE_THREADS = 32
SCRAPE_TEXT_CAP = 4000
