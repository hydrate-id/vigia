import gzip
import io
import json
import logging
import os
import random
import urllib.request
from datetime import date, datetime, timezone

from app.config import (
    BLOCKLIST_CACHE,
    BLOCKLIST_TTL_HOURS,
    CONTENT_CSV,
    DATASET_CSV,
    GAMBLING_URL,
    NRD_DAILY_URL,
    NRD_MANIFEST_URL,
    NRD_SAMPLE,
    RANDOM_STATE,
)
from app.dataset import (
    append_content_batch,
    batch_append_domains,
    read_content_rows,
    read_rows,
    save_content_rows,
)
from ml.features import normalize_domain
from ml.labeling import classify_page
from ml.scrape import fetch_pages

log = logging.getLogger("vigia.enrich")

_UA = {"User-Agent": "Mozilla/5.0 (compatible; Vigia/1.0)"}


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _gambling_set():
    cache = BLOCKLIST_CACHE
    stale = True
    if cache.exists():
        age_h = (datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        stale = age_h > BLOCKLIST_TTL_HOURS
    if not stale:
        with open(cache, encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    data = _get(GAMBLING_URL)
    domains = set()
    for line in io.StringIO(data.decode("utf-8", errors="replace")):
        d = normalize_domain(line.strip())
        if d is not None:
            domains.add(d)
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(domains)))
    os.replace(tmp, cache)
    return domains


def _latest_closed_date(manifest):
    today = datetime.now(timezone.utc).date()
    dates = [d["date"] for d in manifest.get("days", [])]
    for ds in sorted(dates, reverse=True):
        d = date.fromisoformat(ds)
        if d < today:
            return ds
    return None


def _nrd_domains(day):
    data = _get(NRD_DAILY_URL.format(date=day))
    text = gzip.decompress(data).decode("utf-8", errors="replace")
    return text.splitlines()


def _page_to_entry(domain, page):
    if page is None:
        return None, None
    label, status = classify_page(page)
    if label is None or status not in ("gambling", "normal"):
        return None, None
    combo = " ".join(
        x for x in (page.get("title", ""), page.get("meta", ""), page.get("text", "")) if x
    ).strip()
    if not combo:
        return None, None
    return label, combo


def enrich(csv_path=None, content_path=None, domain_predictor=None, content_predictor=None):
    csv_path = csv_path or DATASET_CSV
    content_path = content_path or CONTENT_CSV
    manifest = json.loads(_get(NRD_MANIFEST_URL).decode())
    day = _latest_closed_date(manifest)
    if day is None:
        return {"trained": False, "reason": "no_closed_nrd_day"}

    gambling = _gambling_set()
    dataset_rows = read_rows(csv_path)

    log.info("NRD day %s, sample", day)
    candidates = []
    for line in _nrd_domains(day):
        d = normalize_domain(line.strip())
        if d is not None and d not in gambling:
            candidates.append(d)

    confirmed = [d for d in candidates if d in gambling]
    pos_added = 0
    if confirmed:
        pos_added, _ = batch_append_domains(csv_path, [(d, 1) for d in confirmed])
        log.info("confirmed append: %d", pos_added)

    result = {"day": day, "confirmed_added": pos_added}

    content_rows = read_content_rows(content_path)
    done = set(dataset_rows) | set(content_rows) | set(gambling)
    fresh = [d for d in candidates if d not in done]
    if not fresh:
        result.update({"content_pos": 0, "content_neg": 0, "scraped": 0})
        return result

    rng = random.Random(RANDOM_STATE)
    sample = rng.sample(fresh, min(NRD_SAMPLE, len(fresh)))
    pages = fetch_pages(sample)

    items = []
    for d in sample:
        page = pages.get(d)
        label, combo = _page_to_entry(d, page)
        if label is not None:
            items.append((d, label, combo))
    if not items:
        result.update({"content_pos": 0, "content_neg": 0, "scraped": len(pages)})
        return result

    pos = [it for it in items if it[1] == 1]
    neg = [it for it in items if it[1] == 0]
    pos_added, _ = append_content_batch(content_path, pos)
    neg_added, _ = append_content_batch(content_path, neg)
    log.info("content append: pos=%d neg=%d", pos_added, neg_added)
    result.update(
        {
            "content_pos": pos_added,
            "content_neg": neg_added,
            "scraped": len(pages),
        }
    )
    return result


def bootstrap_content(csv_path=None, content_path=None, per_class=400):
    csv_path = csv_path or DATASET_CSV
    content_path = content_path or CONTENT_CSV
    rows = read_rows(csv_path)
    if not rows:
        return {"trained": False, "reason": "empty_dataset"}
    rng = random.Random(RANDOM_STATE)
    pos_domains = [d for d, v in rows.items() if v == 1]
    neg_domains = [d for d, v in rows.items() if v == 0]
    sample_domains = rng.sample(pos_domains, min(per_class, len(pos_domains))) + rng.sample(
        neg_domains, min(per_class, len(neg_domains))
    )
    pages = fetch_pages(sample_domains)

    new_rows = {}
    for d in sample_domains:
        page = pages.get(d)
        label, combo = _page_to_entry(d, page)
        if label is not None:
            new_rows[d] = (label, combo)

    keep = read_content_rows(content_path)
    for d, (label, combo) in new_rows.items():
        keep[d] = (label, combo)
    save_content_rows(content_path, keep)
    n1 = sum(1 for v in keep.values() if v[0] == 1)
    n0 = len(keep) - n1
    log.info("content bootstrap: scraped=%d labeled=%d total=%d (%d/%d)", len(sample_domains), len(new_rows), len(keep), n1, n0)
    return {"scraped": len(sample_domains), "labeled": len(new_rows), "total": len(keep)}
