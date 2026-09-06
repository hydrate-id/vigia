import json
import threading

import numpy as np
import onnxruntime as ort

from ml.features import normalize_domain


class Predictor:
    def __init__(self, model_path, metrics_path, threshold=0.5, normalizer=None):
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.threshold = threshold
        self.normalizer = normalizer or (lambda s: s)
        self._session = None
        self._meta = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if not self.model_path.exists():
                self._session = None
                self._meta = None
                return False
            session = ort.InferenceSession(str(self.model_path))
            meta = {}
            if self.metrics_path.exists():
                meta = json.loads(self.metrics_path.read_text())
            self._session = session
            self._meta = meta
            return True

    def ready(self):
        return self._session is not None

    def _run(self, value):
        if self._session is None:
            return None
        with self._lock:
            outputs = self._session.run(
                None,
                {self._session.get_inputs()[0].name: np.array([[value]])},
            )
        proba = None
        for name, out in zip(
            [o.name for o in self._session.get_outputs()], outputs
        ):
            if "probab" in name.lower():
                proba = out
                break
        if proba is None:
            proba = outputs[-1]
        if isinstance(proba, (list, tuple)) and proba and isinstance(proba[0], dict):
            return float(proba[0].get(1, 0.0))
        arr = np.asarray(proba)
        if arr.ndim == 2:
            return float(arr[0][1])
        return float(arr[0])

    def p_gambling(self, value):
        return self._run(value)

    def predict(self, raw_domain):
        if self._session is None:
            return None
        domain = self.normalizer(raw_domain)
        if domain is None:
            return None
        p1 = self._run(domain)
        if p1 is None:
            return None
        return self._verdict(domain, p1)

    def score_text(self, text):
        if not text or self._session is None:
            return None
        p1 = self._run(text)
        if p1 is None:
            return None
        return self._verdict(None, p1)

    def _verdict(self, domain, p1):
        is_gambling = p1 >= self.threshold
        confidence = max(p1, 1 - p1) * 100
        verdict = {
            "is_gambling": is_gambling,
            "confidence_percent": round(confidence, 1),
            "probability_gambling": round(p1, 4),
        }
        if domain is not None:
            verdict["domain"] = domain
        return verdict
