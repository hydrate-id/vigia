import gzip
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

from app.config import SCRAPE_MAX_BYTES, SCRAPE_TEXT_CAP, SCRAPE_THREADS, SCRAPE_TIMEOUT

_UA = "Mozilla/5.0 (compatible; Vigia/1.0)"
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_META_DESC_RE = re.compile(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', re.S | re.I)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _decode_bytes(data):
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return data.decode("utf-8", errors="replace")


def _clean(html_bytes):
    doc = _decode_bytes(html_bytes)
    title_m = _TITLE_RE.search(doc)
    title = title_m.group(1).strip() if title_m else ""
    meta_m = _META_DESC_RE.search(doc)
    meta = meta_m.group(1).strip() if meta_m else ""
    raw_lower = doc.lower()
    markers = raw_lower if len(raw_lower) <= SCRAPE_TEXT_CAP else raw_lower
    parser = _TextExtractor()
    parser.feed(doc)
    text = " ".join(" ".join(parser.parts).split())
    return {
        "title": title[:500],
        "meta": meta[:500],
        "text": text[:SCRAPE_TEXT_CAP] if text else "",
        "raw_markers": markers,
        "html_len": len(doc),
    }


def _fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=SCRAPE_TIMEOUT) as resp:
            data = resp.read(SCRAPE_MAX_BYTES + 1)
            if resp.headers.get("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
            return data[:SCRAPE_MAX_BYTES]
    except Exception:
        return None


def fetch_page(domain):
    """Fetch a domain page or None."""
    for scheme in ("https", "http"):
        data = _fetch_bytes(f"{scheme}://{domain}/")
        if data:
            page = _clean(data)
            if page["title"] or page["text"]:
                page["domain"] = domain
                return page
    return None


def fetch_text(domain):
    page = fetch_page(domain)
    if page is None:
        return None
    text = page["text"]
    return text if text else None


def fetch_pages(domains):
    results = {}
    with ThreadPoolExecutor(max_workers=SCRAPE_THREADS) as ex:
        futures = {ex.submit(fetch_page, d): d for d in domains}
        for fut, d in futures.items():
            results[d] = fut.result()
    return results
