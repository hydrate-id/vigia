import gzip
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

from app.config import SCRAPE_MAX_BYTES, SCRAPE_TEXT_CAP, SCRAPE_THREADS, SCRAPE_TIMEOUT

SCRAPINGBEE_URL = "https://app.scrapingbee.com/api/v1/"

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
    def __init__(self, ssl_verify=True):
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
        with self.opener.open(req, timeout=SCRAPE_TIMEOUT) as resp:
            data = resp.read(SCRAPE_MAX_BYTES + 1)
            final_url = resp.geturl()
            if resp.headers.get("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
            return data[:SCRAPE_MAX_BYTES], final_url


_VERIFY = _Opener()
_NO_VERIFY = _Opener(ssl_verify=False)


def _fetch_bytes(url):
    for opener in (_VERIFY, _NO_VERIFY):
        try:
            return opener.fetch(url)
        except Exception:
            continue
    return None, None


def _fetch_scrapingbee(url):
    token = os.environ.get("SCRAPINGBEE_TOKEN")
    if not token:
        return None
    params = {
        "api_key": token,
        "url": url,
        "render_js": "true",
        "premium_proxy": "true",
        "country_code": "id",
    }
    sb_url = SCRAPINGBEE_URL + "?" + urllib.parse.urlencode(params)
    try:
        return _VERIFY.fetch(sb_url)
    except Exception:
        return None, None


def fetch_page(domain):
    """Fetch a domain page or None. Follows redirects with a browser UA."""
    for scheme in ("https", "http"):
        data, final_url = _fetch_bytes(f"{scheme}://{domain}/")
        if not data:
            sb = _fetch_scrapingbee(f"{scheme}://{domain}/")
            if sb:
                data, final_url = sb
        if not data:
            continue
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
