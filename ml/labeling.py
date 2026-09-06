import re

GAMBLING_TERMS = [
    # indonesian
    "togel", "judol", "judi online", "bandar judi", "situs judi", "slot gacor",
    "slot online", "sbobet", "maxbet", "agen slot", "daftar slot", "taruhan bola",
    "bookie", "bandarqq", "judi bola", "situs slot", "tembak ikan",
    # english / generic
    "casino", "slot machine", "online casino", "betting", "bookmaker",
    "roulette", "blackjack", "baccarat", "jackpot", "wager", "sportsbook",
    "sport betting", "parlay", "casino online", "slots",
    # chinese
    "老虎机", "赌场", "百家乐", "娱乐城", "彩票", "下注", "博彩", "轮盘",
    "投注", "开奖", "时时彩", "六合彩", "真人视讯", "棋牌", "电子游艺",
    "捕鱼游戏", "线上赌场", "真钱游戏", "注册送", "提现秒到", "澳门赌场",
    "体育博彩", "首充双倍",
    # chinese casino euphemisms / slogans
    "天生赢家", "顶级娱乐", "娱乐平台", "真人娱乐", "返水", "大额无忧",
    "实力直营", "信誉平台", "高额返水", "娱乐官网", "博狗", "新葡京",
    # thai
    "บาคาร่า", "สล็อต", "คาสิโน", "เดิมพัน", "แทงบอล", "เว็บพนัน",
    "คาสิโนออนไลน์", "เกมยิงปลา", "หวย",
    # korea / others
    "카지노", "슬롯", "바카라", "베팅",
]

# neutral markers that indicate a parked / auction / registrar holding page
PARKING_MARKERS = [
    "domain is for sale", "for sale", "this domain", "domain name has been",
    "registered with", "buy this domain", "auctions", "see all auctions",
    "namecheap", "gandi", "godaddy", "sedo", "afternic", "bodis",
    "want a domain name like this", "expired domain",
]

# anti-bot / JS-challenge markers — only meaningful when page has no real text
CHALLENGE_MARKERS = [
    "jsjiami", "sm4.js", "checking your browser", "cf-chl", "cf-challenge",
    "one moment, please", "加载中", "please wait while your request",
    "anbieterpruefung", "dns-verify", "crypto-loot", "verify you are human",
]
# obfuscator watermark that signals JS-shell gambling pages even w/o challenge text
SHELL_JS_MARKERS = ["jsjiami", "sm4.js"]

_LOGIN_TITLES = ["login", "log in", "sign in", "member login", "登錄", "登录", "登入"]

# real <script src="...sm4.js">, SM4 anti-bot lib used by Chinese gambling shells
_SM4_SCRIPT_RE = re.compile(r'src=["\'][^"\']*sm4\.js["\']', re.I)

# CJK + Thai detection for shell-title signal
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")

# Chinese gambling-shell title terms
_ZH_GAMBLING_SHELL = [
    "澳门", "赌", "博彩", "娱乐", "彩票", "彩民", "老虎机", "百家乐",
    "棋牌", "电竞", "体育投注", "信息站",
]

_JUNK_TLDS = (".sbs", ".vip", ".top", ".xyz", ".cc", ".icu", ".fun", ".site", ".club", ".online", ".live", ".bet", ".win", ".pro")

_MIN_REAL_TEXT = 25


def _lower_combine(title, text, meta):
    return " ".join(x or "" for x in (title, text, meta)).lower()


def _hit(text, terms):
    for t in terms:
        if t in text:
            return True
    return False


def _real_text_len(title, text, meta):
    return len(((title or "") + " " + (text or "")).strip())


def _numeric_prefix(domain):
    base = domain.split(":")[0].lower()
    return base[:1].isdigit()


def is_shell(title, text, meta, raw_markers=None):
    combined = _lower_combine(title, text, meta)
    real_len = _real_text_len(title, text, meta)
    if real_len >= _MIN_REAL_TEXT:
        return False
    if _hit(combined, CHALLENGE_MARKERS):
        return True
    if raw_markers and _hit((raw_markers or "").lower(), SHELL_JS_MARKERS):
        return True
    if text and _hit(combined, CHALLENGE_MARKERS):
        return True
    if not text:
        t = (title or "").lower()
        if _hit(t, _LOGIN_TITLES) or "信息站" in (title or ""):
            return True
    return False


def is_gambling_shell(title, text, meta, raw_markers, domain=""):
    """Detect gambling JS-shell (anti-bot + CJK disguise title).

    Used when page content is JS-only. Fires when anti-bot marker present AND
    (CJK gambling title OR numeric junk domain).
    """
    if not is_shell(title, text, meta, raw_markers):
        return False
    raw = (raw_markers or "").lower()
    has_ab = bool(_SM4_SCRIPT_RE.search(raw or "")) or "jsjiami" in raw
    if not has_ab:
        return False
    title_zh_gambling = _hit((title or ""), _ZH_GAMBLING_SHELL)
    domain_junk = _numeric_prefix(domain) and domain.lower().endswith(_JUNK_TLDS)
    if title_zh_gambling and (_CJK_RE.search(title or "") or _THAI_RE.search(title or "")):
        return True
    return domain_junk and _CJK_RE.search(title or "")


def is_parking(title, text, meta):
    return _hit(_lower_combine(title, text, meta), PARKING_MARKERS)


def _gambling_hits(title, text, meta):
    combined = _lower_combine(title, text, meta)
    return sum(1 for t in GAMBLING_TERMS if t in combined)


def is_gambling_text(title, text, meta, real_len=None):
    real_len = _real_text_len(title, text, meta) if real_len is None else real_len
    hits = _gambling_hits(title, text, meta)
    if hits >= 2:
        return True
    if hits == 1 and real_len < 80:
        return True
    return False


def classify_page(page):
    """Label a page. Returns (label, status).

    label: 1 = gambling, 0 = normal.
    status explains when label is None.
    """
    title = page.get("title") or ""
    text = page.get("text") or ""
    meta = page.get("meta") or ""
    raw = page.get("raw_markers")
    domain = page.get("domain") or ""
    if is_gambling_shell(title, text, meta, raw, domain):
        return 1, "gambling_shell"
    if is_shell(title, text, meta, raw):
        return None, "shell"
    if is_parking(title, text, meta):
        return None, "parked"
    real_len = _real_text_len(title, text, meta)
    if is_gambling_text(title, text, meta, real_len):
        return 1, "gambling"
    if real_len >= _MIN_REAL_TEXT:
        return 0, "normal"
    return None, "ambiguous"
