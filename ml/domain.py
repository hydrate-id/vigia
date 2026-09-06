import re

_DOMAIN_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$"
)
_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def normalize_domain(raw):
    if raw is None:
        return None
    s = str(raw).strip().lower()
    s = re.sub(r"^[a-z][a-z0-9+.-]*://", "", s)
    s = s.split("/", 1)[0]
    s = s.split("?", 1)[0].split("#", 1)[0]
    s = re.sub(r":\d+$", "", s)
    s = s.strip(".")
    if s.startswith("www."):
        s = s[4:]
    if not s or len(s) > 253:
        return None
    if "." not in s:
        return None
    if not _DOMAIN_RE.fullmatch(s):
        return None
    if _IP_RE.fullmatch(s):
        return None
    return s
