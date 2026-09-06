import os
import time
from collections import defaultdict, deque

RATE_LIMIT = 3
RATE_WINDOW = 1.0

_buckets = defaultdict(deque)
_last_cleanup = time.time()


def unlimited_token(token):
    configured = os.environ.get("API_TOKEN", "")
    return bool(configured) and token == configured


def client_ip(request):
    for header in ("x-vercel-forwarded-for", "x-forwarded-for", "x-real-ip"):
        value = request.headers.get(header)
        if value:
            first = value.split(",")[0].strip()
            if first:
                return first
    return "unknown"


def allow(ip):
    now = time.time()
    global _last_cleanup
    if now - _last_cleanup > 60:
        _buckets.clear()
        _last_cleanup = now
    q = _buckets[ip]
    while q and now - q[0] >= RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return False
    q.append(now)
    return True
