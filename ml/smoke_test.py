import tempfile
from pathlib import Path

from app.config import MODELS_DIR
from app.dataset import save_content_rows, save_rows
from app.predictor import Predictor
from ml.domain import normalize_domain
from ml.labeling import classify_page
from ml.train import retrain_content, retrain_domain

# Synthetic domains for DOMAIN-PREDICTOR smoke only (not content, not live sites).
# These names are fictional fixtures on reserved TLDs (.invalid / .test / .example)
# that do not resolve on the public Internet. When this smoke test was written,
# none of these domains existed as real sites. They exist solely to exercise the
# domain model on gambling-like lexical patterns — not to promote, list, or
# advertise any real online gambling operator.
JUDOL = [
    "slot-gacor-maxwin777.invalid",
    "situs-slot-gacor-online.invalid",
    "rtp-slot-gacor-hari-ini.invalid",
    "agen-slot-gacor-88.invalid",
    "judi-online-slot-gacor.invalid",
    "deposit-pulsa-slot-gacor.invalid",
    "pragmatic-play-slot-gacor.invalid",
    "gacor-slot-maxwin.invalid",
    "slotgacor777maxwin.invalid",
    "situsjudislotgacor.invalid",
    "slot-gacor-terpercaya-88.invalid",
    "maxwin-slot-gacor-hari-ini.invalid",
    "daftar-slot-gacor-online.invalid",
    "link-alternatif-slot-gacor.invalid",
    "bocoran-rtp-slot-gacor.invalid",
    "slot-gacor-maxwin777.test",
    "rtp-slot-gacor-hari-ini.example",
    "agen-slot-gacor-88.example",
]
NORMAL = [
    "google.com", "facebook.com", "wikipedia.org", "github.com", "tokopedia.com",
    "shopee.co.id", "detik.com", "kompas.com", "cimbniaga.co.id", "bca.co.id",
    "lazada.co.id", "bukalapak.com", "youtube.com", "twitter.com", "netflix.com",
    "blibli.com", "gojek.com", "grab.com", "cloudflare.com", "mozilla.org",
]

# Synthetic fixtures for inference priors only (reserved TLDs; not real operators).
# LEXICON: clear gambling vocabulary that the n-gram alone often misses.
# WEIRD: random-looking junk-TLD hosts with no gambling words in the name.
LEXICON_FIXTURES = [
    "judol-test-fixture.invalid",
    "togel4d-fixture.invalid",
    "situs-judi-slot.invalid",
    "agenjudi88.invalid",
    "bandarqq99.invalid",
]
WEIRD_FIXTURES = [
    "a8f3k2m9p1q7.xyz",
    "009951fixture.sbs",
    "x7k9m2p4q8n1.vip",
]
PRIOR_NEGATIVES = [
    "google.com",
    "github.com",
    "judicial.org",
    "slot2.akamai.net",
]

JUDOL_TEXT = [
    "在线赌场 老虎机 真钱游戏 立即存款 提款秒到",
    "老虎机 在线娱乐城 投注 注册送彩金 百家乐",
    "situs slot gacor hari ini daftar deposit pulsa tanpa potongan maxwin",
    "บาคาร่า คาสิโน สล็อต เล่นได้ตลอด 24 ชั่วโมง ฝากถอนไว",
    "百家乐 轮盘 骰宝 真人荷官 下注赢大奖 优惠活动",
    "สล็อตแตกง่าย เกมยิงปลา รับโบนัส สมัครสมาชิกตอนนี้เลย",
    "judi online terpercaya togel 4d pasaran lengkap bonus new member",
    "娱乐城 注册即送 天天返水 高水位 信誉平台",
    "slot deposit dana 10k bonus 100 persen gacor hari ini daftar",
    "poker domino qiu qiu uang asli terbaik indonesia jackpot besar",
    "老虎机游戏 投注平台 澳门赌场 线上娱乐 信誉最好 提现快",
    "baccarat casino live dealer game online real money withdraw",
    "สล็อตเว็บตรง ไม่ผ่านเอเย่นต์ แตกง่าย โบนัสเยอะ",
    "daftar situs betting bola parlay online resmi no 1 indonesia",
    "欢乐麻将 捕鱼游戏 电子游艺 首充双倍 大额无忧",
    "กีฬาเดิมพัน คาสิโนสด สล็อต pg ทางเข้าเล่นล่าสุด",
]
NORMAL_TEXT = [
    "Welcome to our site, read the latest technology news and product reviews.",
    "Contoh artikel berita nasional tentang ekonomi dan olahraga hari ini.",
    "Learn python programming with tutorials for beginners and experts.",
    "Recipe for healthy breakfast with oats fruits and yogurt in minutes.",
    "今日天气预报 晴 气温25度 适合外出活动",
    "公司简介 我们的团队专注于软件开发和产品设计",
    "The quick brown fox jumps over the lazy dog near the river.",
    "Resep masakan rumahan sederhana ayam goreng dan sambal terasi.",
    "Read the documentation before installing the software package.",
    "Conference schedule for next week includes keynote and workshops.",
    "บทความเกี่ยวกับการดูแลสุขภาพ และการออกกำลังกายประจำวัน",
    "Product specification sheet with dimensions weight and warranty info.",
    "学校新闻 开学典礼 时间安排 学生名单公布",
    "Open source community guidelines and contribution checklist.",
    "Weekly newsletter with curated links about science and nature.",
    "Contact us page with office address phone and email form.",
]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="vigia_smoke_"))
    csv_path = tmp / "dataset.csv"
    content_path = tmp / "content.csv"
    models_dir = tmp / "models"

    rows = {d: 1 for d in JUDOL}
    rows.update({d: 0 for d in NORMAL})
    save_rows(csv_path, rows)
    # Synthetic placeholder domains for content rows only (never real operators).
    crows = {d: (1, t) for d, t in zip([f"fixture-pos-{d}.invalid" for d in range(len(JUDOL_TEXT))], JUDOL_TEXT)}
    crows.update({d: (0, t) for d, t in zip([f"fixture-neg-{d}.invalid" for d in range(len(NORMAL_TEXT))], NORMAL_TEXT)})
    save_content_rows(content_path, crows)

    m_domain = retrain_domain(csv_path, models_dir)
    assert m_domain["trained"], f"domain retrain failed: {m_domain}"
    print(f"OK retrain domain: acc={m_domain['accuracy']} n_train={m_domain['n_train']}")

    m_content = retrain_content(content_path, models_dir)
    assert m_content["trained"], f"content retrain failed: {m_content}"
    print(f"OK retrain content: acc={m_content['accuracy']} n_train={m_content['n_train']}")

    assert (models_dir / "model.onnx").exists(), "model.onnx missing"
    assert (models_dir / "model_content.onnx").exists(), "model_content.onnx missing"

    dp = Predictor(models_dir / "model.onnx", models_dir / "metrics.json", normalizer=normalize_domain)
    assert dp.load(), "load domain onnx failed"

    cp = Predictor(models_dir / "model_content.onnx", models_dir / "metrics_content.json")
    assert cp.load(), "load content onnx failed"

    # JUDOL fixtures are domain-predictor-only: every positive must score true.
    for d in JUDOL:
        r = dp.predict(d)
        assert r is not None and r["is_gambling"] is True, f"JUDOL domain must be true: {d} -> {r}"
    hits = len(JUDOL)
    for d in NORMAL:
        r = dp.predict(d)
        assert r is not None, f"predict {d} None"
        if r["is_gambling"] is False:
            hits += 1
    print(f"OK onnx predict domain: {hits}/{len(JUDOL) + len(NORMAL)} correct")
    assert hits >= 28, "domain smoke accuracy too low"

    # Shipped production domain model must also flag every JUDOL fixture true.
    prod = Predictor(MODELS_DIR / "model.onnx", MODELS_DIR / "metrics.json", normalizer=normalize_domain)
    if prod.load():
        for d in JUDOL:
            r = prod.predict(d)
            assert r is not None and r["is_gambling"] is True, (
                f"production domain model must flag JUDOL fixture true: {d} -> {r}"
            )
        print(f"OK production domain model: {len(JUDOL)}/{len(JUDOL)} JUDOL fixtures true")

        for d in LEXICON_FIXTURES:
            r = prod.predict(d)
            assert r is not None and r["is_gambling"] is True and r.get("domain_prior") == "lexicon", (
                f"lexicon prior must flag {d} -> {r}"
            )
        for d in WEIRD_FIXTURES:
            r = prod.predict(d)
            assert r is not None and r["is_gambling"] is True and r.get("domain_prior") == "weird", (
                f"weird prior must flag {d} -> {r}"
            )
        for d in PRIOR_NEGATIVES:
            r = prod.predict(d)
            assert r is not None and r["is_gambling"] is False and "domain_prior" not in r, (
                f"prior must not FP {d} -> {r}"
            )
        print(
            f"OK domain priors: lexicon={len(LEXICON_FIXTURES)} "
            f"weird={len(WEIRD_FIXTURES)} neg={len(PRIOR_NEGATIVES)}"
        )

    chits = 0
    for t, expect in [(t, True) for t in JUDOL_TEXT] + [(t, False) for t in NORMAL_TEXT]:
        r = cp.score_text(t)
        assert r is not None, "score_text None"
        if r["is_gambling"] == expect:
            chits += 1
    print(f"OK onnx predict content: {chits}/{len(JUDOL_TEXT) + len(NORMAL_TEXT)} correct")
    assert chits >= len(JUDOL_TEXT) + len(NORMAL_TEXT) - 2, "content smoke accuracy too low"

    assert dp.predict("not a domain") is None, "invalid domain should be None"
    print("OK invalid domain -> None")

    # classify_page fixtures use synthetic domain placeholders only (no real operators).
    cases = [
        ({"title": "Situs Togel Terpercaya Fixture", "meta": "", "text": "slot online gacor daftar sekarang"}, 1, "gambling"),
        ({"title": "老虎机 赌场 注册送彩金", "meta": "", "text": "百家乐 轮盘 投注"}, 1, "gambling"),
        ({"title": "Home", "meta": "portal berita ekonomi", "text": "Berita nasional dan internasional hari ini lengkap."}, 0, "normal"),
        ({"title": "澳门信息站", "meta": "", "text": ""}, None, "shell"),
        ({"title": "Domain is for sale", "meta": "", "text": "Buy this domain at sedo auction."}, None, "parked"),
        ({"title": "Site", "meta": "", "text": "加载中"}, None, "shell"),
        ({"title": "澳门信息站", "meta": "", "text": "", "domain": "fixture-gambling-shell.invalid", "raw_markers": '<script src="js/sm4.js"></script>'}, 1, "gambling_shell"),
        ({"title": "搜狐", "meta": "", "text": "搜狐门户 新闻 体育 彩票 健康 财经 娱乐 科技 汽车 房产 教育 文化 时尚 视频 军事 旅游 母婴 星座 奥运 游戏 邮箱 博客 搜狐号 24小时直播 大视野 公益 畅游 17173 政务 网络监督专区 欢迎监督 如实举报 联系我们 法律声明 隐私权政策 网站地图 帮助中心", "domain": "sohu.com", "raw_markers": "..."}, 0, "normal"),
        ({"title": "Hashed", "meta": "", "text": "python sdk docs install guides", "domain": "example.com", "raw_markers": "mrktep6q3wyh7dti%252brmksm4nbivqwjr1"}, 0, "normal"),
        ({"title": "Fixture凯发·天生赢家", "meta": "", "text": "亚洲顶级在线娱乐平台 官方实力直营 信誉保障 大额无忧", "domain": "fixture-zh-casino.invalid", "raw_markers": "<html lang=cn>"}, 1, "gambling"),
        ({"title": "首页", "meta": "", "text": "欢迎访问本站 新闻资讯 产品中心 关于我们 联系我们 公司介绍 人才招聘 服务支持 常见问题 隐私政策", "domain": "example.cn", "raw_markers": "..."}, 0, "normal"),
        ({"title": "APP下载", "meta": "", "text": "APP下载 全网独家 日入300-3000 加入服务器 bc999 添加账号 S88888 接待老师 注册教程 下载APP 注册并登录", "domain": "fixture-recruit-funnel.invalid", "raw_markers": "<html lang=zh-cn>"}, 1, "gambling_funnel"),
        ({"title": "加入服务器", "meta": "", "text": "欢迎加入我们的游戏服务器 输入IP地址 开始游玩 服务器状态 常见问题 关于我们 联系我们 下载客户端", "domain": "mc-server.com", "raw_markers": "..."}, 0, "normal"),
        ({"title": "Job", "meta": "", "text": "兼职招聘 日入200-500 立即报名 联系方式 公司简介 岗位要求 福利待遇 上班时间 工作地点", "domain": "job.cn", "raw_markers": "..."}, 0, "normal"),
    ]
    for page, elabel, estatus in cases:
        label, status = classify_page(page)
        assert label == elabel and status == estatus, f"classify {page}: {label}/{status}, expect {elabel}/{estatus}"
    print(f"OK classify_page (no-FP): {len(cases)}/{len(cases)}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
