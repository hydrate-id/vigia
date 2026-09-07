import gzip
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

from app.config import SCRAPE_MAX_BYTES, SCRAPE_TEXT_CAP, SCRAPE_THREADS, SCRAPE_TIMEOUT

SCRAPINGANT_URL = "https://api.scrapingant.com/v2/general"
SCRAPINGANT_TIMEOUT = 45
# Budget guard: stop calling the paid fallback once this many credits burned
# in a single process run (default 250/night -> ~7.5k of the 10k free month).
SA_BUDGET_CREDITS = int(os.environ.get("SCRAPE_ANT_BUDGET_CREDITS", "250"))
_sa_lock = threading.Lock()
_sa_credits = 0

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
_META_DESC_RE = re.compile(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', re.S | re.I)


class _Redirect(urllib.request.HTTPRedirectHandler):
    """Keep request headers when following cross-host redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        for k, v in req.headers.items():
            new.add_unredirected_header(k, v)
        return new


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
    parser = _TextExtractor()
    parser.feed(doc)
    text = " ".join(" ".join(parser.parts).split())
    return {
        "title": title[:500],
        "meta": meta[:500],
        "text": text[:SCRAPE_TEXT_CAP] if text else "",
        "raw_markers": doc[:SCRAPE_TEXT_CAP].lower(),
        "html_len": len(doc),
        "final_url": "",
    }


class _Opener:
    def __init__(self, ssl_verify=True, timeout=SCRAPE_TIMEOUT):
        self.timeout = timeout
        ctx = ssl.create_default_context() if ssl_verify else ssl._create_unverified_context()
        handlers = [_Redirect(), urllib.request.HTTPSHandler(context=ctx)]
        self.opener = urllib.request.build_opener(*handlers)

    def fetch(self, url):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,id;q=0.7",
                "Accept-Encoding": "gzip",
            },
        )
        with self.opener.open(req, timeout=self.timeout) as resp:
            data = resp.read(SCRAPE_MAX_BYTES + 1)
            final_url = resp.geturl()
            headers = resp.headers
            if resp.headers.get("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
            return data[:SCRAPE_MAX_BYTES], final_url, headers


_VERIFY = _Opener()
_NO_VERIFY = _Opener(ssl_verify=False)
_SA_OPENER = _Opener(timeout=SCRAPINGANT_TIMEOUT)


def _fetch_bytes(url):
    for opener in (_VERIFY, _NO_VERIFY):
        try:
            data, final_url, _headers = opener.fetch(url)
            return data, final_url
        except Exception:
            continue
    return None, None


def fallback_credits_used():
    return _sa_credits


def _fetch_scrapingant(url):
    """Render url via ScrapingAnt (JS + datacenter proxy). None, None when spent."""
    global _sa_credits
    key = os.environ.get("SCRAPINGANT_API_KEY")
    if not key:
        return None, None
    # free tier allows one concurrent request; serialize + gate on budget here
    with _sa_lock:
        if _sa_credits >= SA_BUDGET_CREDITS:
            return None, None
        params = {
            "url": url,
            "x-api-key": key,
            "browser": "true",
            "proxy_country": "id",
        }
        ant_url = SCRAPINGANT_URL + "?" + urllib.parse.urlencode(params)
        for _attempt in range(3):
            try:
                data, final_url, headers = _SA_OPENER.fetch(ant_url)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(2)
                    continue
                return None, None
            except Exception:
                return None, None
            break
        else:
            return None, None
        try:
            cost = int(headers.get("Ant-credits-cost") or 0)
        except (TypeError, ValueError):
            cost = 0
        _sa_credits += cost
        if not data:
            return None, None
        return data, final_url


def fetch_page(domain):
    """Fetch a domain page or None. Follows redirects with a browser UA."""
    for scheme in ("https", "http"):
        data, final_url = _fetch_bytes(f"{scheme}://{domain}/")
        if data:
            page = _clean(data)
            page["final_url"] = final_url or ""
            if page["title"] or page["text"]:
                page["domain"] = domain
                return page
    data, final_url = _fetch_scrapingant(f"https://{domain}/")
    if data:
        page = _clean(data)
        page["final_url"] = final_url or ""
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
