"""叁 · 先驱（96 BPM 在前 44 拍渐快到 112 BPM）。左下角年份标尺。

章节卡 → 万户（点火、椅下火箭助推、离地飞出画面、尾烟转场）→ 1783 热气球 → 1853 凯利 → 1865 凡尔纳 →
李林塔尔 → 1903 分屏 → 1920 纽约时报 → 1926 戈达德 → 1931 皮卡尔 → 1946 V-2 → 1947 X-1 音爆 → 一拍静音。
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from engine.core import (W, H, FPS, CREAM, INK, GOLD, PAPER, PAPER_DARK, SMOKE, chapter_card, post, place, lerp,
                         ease_io, ease_in, ease_out, ease_out_expo, clamp01, glow, caption, duotone, sharpen,
                         cover_scale, draw_text, draw_img, motion_blur, flash_at, shake_at, smoke_wipe, fbm, xx, yy,
                         YearTicker, font)
from engine.film import Chapter, Shot
from engine.media import image
from engine.shots import ken, whip, split
from engine import music as M
from chapters.wanhu_art import build as wanhu_build
from chapters.newsprint import newspaper

SEPIA_D, SEPIA_L = (0.09, 0.06, 0.04), (0.94, 0.88, 0.76)
SMOKE_COL = np.array([0.60, 0.54, 0.48], np.float32)
FIRE = np.array([1.0, 0.48, 0.12], np.float32)


def sepia(img, contrast=1.25, gamma=1.08):
    return sharpen(duotone(img, SEPIA_D, SEPIA_L, gamma, contrast), 0.5)


_CACHE = {}


def src(key):
    """先驱章用到的图，统一转成棕褐双色调并缓存。"""
    if key not in _CACHE:
        if key.startswith("verne"):
            boxes = {"01": (0.39, 0.26, 0.86, 0.74), "02": (0.37, 0.27, 0.86, 0.74),
                     "03": (0.14, 0.30, 0.61, 0.82), "04": (0.38, 0.29, 0.86, 0.78)}
            cand = key[-2:]
            x = image("PD-04", cand, 2400)
            h, w = x.shape[:2]
            x0, y0, x1, y1 = boxes[cand]
            x = np.ascontiguousarray(np.rot90(x[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], 1))
            _CACHE[key] = sepia(x, 1.15)
        else:
            aid, cand, crop = {"balloon": ("PD-08", "01", None), "wright": ("PD-11", "01", None),
                               "lil1": ("PD-10", "01", (0.07, 0.08, 0.93, 0.80)),
                               "lil2": ("PD-10", "02", (0.12, 0.15, 0.66, 0.60)),
                               "goddard": ("NA-06", "01", None), "v2": ("NA-19", "01", None),
                               "x1": ("NA-20", "01", (0, 0, 1, 0.66)), "capsule": ("NA-05", "01", None)}[key]
            x = image(aid, cand, 3000)
            if crop:
                h, w = x.shape[:2]
                x = x[int(crop[1] * h):int(crop[3] * h), int(crop[0] * w):int(crop[2] * w)]
            _CACHE[key] = sepia(x, 1.4 if key == "v2" else 1.25)
    return _CACHE[key]


def s_card(c):
    return chapter_card(c, "先驱")


# ---------------- 万户 ----------------

WANHU = wanhu_build()
NOZ_ALL = np.array(WANHU["nozzles"], np.float32)
NOZ = NOZ_ALL[NOZ_ALL[:, 1] > 840]
NOZ_CX, NOZ_BOTTOM = float(NOZ[:, 0].mean()), float(NOZ[:, 1].max())


def _along(pts, p):
    seg = [np.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1)]
    dist = p * sum(seg)
    for k, L in enumerate(seg):
        if dist <= L or k == len(seg) - 1:
            q = min(dist / L, 1)
            return lerp(pts[k][0], pts[k + 1][0], q), lerp(pts[k][1], pts[k + 1][1], q)
        dist -= L


def _spark(frame, x, y, fi, size=16, strength=1.4):
    frame += glow(x, y, size, FIRE, strength * (0.8 + 0.2 * np.sin(fi * 2.7)))
    r = np.random.default_rng(fi * 13 + int(x))
    for _ in range(7):
        cv2.circle(frame, (int(x + r.normal(0, size * 1.3)), int(y + r.normal(-size * 0.5, size))), 2, (1.0, 0.82, 0.45), -1)
    return frame


def _flames(dy, power, fi, subset=1.0):
    small = np.zeros((H // 2, W // 2, 3), np.float32)
    r = np.random.default_rng(fi)
    for x, y in NOZ:
        if r.random() > subset:
            continue
        L = (120 + 70 * r.random()) * power
        if L < 2:
            continue
        cx, cy = x / 2, (y + dy) / 2 + L / 4
        wd = max(1, int(7 * power ** 0.5))
        cv2.ellipse(small, (int(cx), int(cy)), (wd, max(1, int(L / 4))), 0, 0, 360, tuple(map(float, FIRE)), -1)
        cv2.ellipse(small, (int(cx), int(cy - L / 10)), (max(1, wd // 2), max(1, int(L / 7))), 0, 0, 360, (1.0, 0.92, 0.65), -1)
    big = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5), (W, H), interpolation=cv2.INTER_LINEAR) * 1.6
    gy = NOZ_BOTTOM + dy + 50
    g = np.exp(-(((xx - NOZ_CX) ** 2) / (2 * 260 ** 2) + ((yy - gy) ** 2) / (2 * 150 ** 2)))[..., None]
    plume = np.exp(-(((xx - NOZ_CX) ** 2) / (2 * 110 ** 2)) - np.clip(yy - gy, 0, None) / (260 * max(power, 0.05)))
    plume = (plume * (yy > gy - 40))[..., None]
    return big + g * FIRE * 0.55 * min(power, 1.2) + plume * (FIRE * 0.7 + 0.25) * min(power, 1.0)


def _ground_smoke(level, rise, lit):
    n = np.roll(SMOKE[0], -int(rise) % H, axis=0) * 0.7 + np.roll(SMOKE[1], -int(rise * 0.6) % H, axis=0) * 0.3
    m = np.clip((yy - (H - level)) / 160 + (n - 0.5) * 1.8, 0, 1)[..., None]
    col = SMOKE_COL + FIRE * 0.45 * lit * np.exp(-((xx - NOZ_CX) ** 2) / (2 * 420 ** 2))[..., None]
    return m, col


def s_wanhu(c):
    """引信 → 喷口冒火星 → 第 8 拍点火（椅下火箭向下喷焰）→ 第 9 拍离地飞出画面 → 第 11.4 拍起尾烟盖满画面。"""
    L = c.lb
    rb = min(L, 7.5) if L < 8 else L
    frame = WANHU["sky"].copy()
    dx = dy = 0.0
    power = 0.0
    if L >= 8:
        power = min(1.0, 0.55 + c.since(8) * 1.2) * (0.9 + 0.1 * np.sin(c.fi * 3.1))
        jr = np.random.default_rng(c.fi)
        dx, dy = jr.normal(0, 2.5), jr.normal(0, 2.5)
        if L >= 9:
            dy += -1500 * ease_in((L - 9) / 2.4, 2.3)
    fly_rgb, fly_a = WANHU["fly_rgb"], WANHU["fly_a"]
    if dx or dy:
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        fly_rgb = cv2.warpAffine(fly_rgb, m, (W, H))
        fly_a = cv2.warpAffine(fly_a[..., 0], m, (W, H))[..., None]
    if power > 0:
        rim = np.exp(-(((xx - NOZ_CX) ** 2) / (2 * 300 ** 2) + ((yy - (NOZ_BOTTOM + dy - 60)) ** 2) / (2 * 140 ** 2)))[..., None]
        fly_rgb = fly_rgb + rim * FIRE * 0.55 * power
    frame = frame * (1 - fly_a) + fly_rgb * fly_a
    if L < 8:
        p = clamp01((rb - 0.3) / 4.5)
        if p < 1:
            x, y = _along(WANHU["fuse"], p)
            _spark(frame, x, y, c.fi)
        elif rb < 5.6:
            q = clamp01((rb - 4.8) / 0.8)
            hub = WANHU["fuse"][-1]
            for bx, by in WANHU["branches"]:
                _spark(frame, lerp(hub[0], bx, q), lerp(hub[1], by, q), c.fi, 12, 1.1)
        else:
            k = clamp01((rb - 5.6) / 1.9)
            frame = frame + _flames(0, 0.12 + 0.25 * k, c.fi, subset=0.25 + 0.5 * k)
    else:
        frame = frame + _flames(dy, power, c.fi)
    frame = frame * (1 - WANHU["gnd_a"]) + WANHU["gnd_rgb"] * WANHU["gnd_a"]
    zoom = lerp(1.0, 1.12, ease_io(rb / 8)) if L < 8 else 1.12
    cam_y = 600 + (0.12 * dy if L >= 9 else 0)
    m = cv2.getRotationMatrix2D((1262.0, float(cam_y)), 0, zoom)
    m[0, 2] += W / 2 + 300 - 1262
    m[1, 2] += H / 2 + 60 - cam_y
    frame = cv2.warpAffine(frame, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    caption(frame, c, 0.5, [("万户", "big"), ("47 支火箭，一把椅子", "mid"),
                            ("今天，月球背面有一座以他命名的环形山", "small")], scrim=0.5)
    if L >= 8:
        level = lerp(60, 380, ease_io((L - 8) / 3.4)) if L < 11.4 else lerp(380, H + 400, ease_in((L - 11.4) / 0.6, 1.6))
        sm, col = _ground_smoke(level, c.since(8) * 70, power)
        frame = frame * (1 - sm) + col * sm
    elif L >= 7.5:
        frame = frame * 0.8
    shake = (0.0, 0.0)
    if 8 <= L < 11.4:
        sr = np.random.default_rng(c.fi + 99)
        amp = 9 * (1 - 0.6 * clamp01((L - 9) / 2.4))
        shake = (sr.normal(0, amp), sr.normal(0, amp))
    return post(frame, c.fi, shake=shake, grain=0.035)


# ---------------- 1783 热气球 ----------------

def s_balloon(c):
    s_ = src("balloon")
    h, w = s_.shape[:2]
    k = ease_io(c.k)
    base = 1280 / max(h, w)
    scale = 4.2 * base * ((0.80 / 4.2) ** k)
    cx, cy = lerp(0.53 * w, 0.5 * w, k), lerp(0.35 * h, 0.47 * h, k)
    rgb, a = place(s_, cx, cy, scale, lerp(0, -1.2, k), lerp(W / 2, 1210, k), H / 2)
    frame = PAPER_DARK * 1.6 * (1 - a) + rgb * a
    frame = frame * (1 - 0) - 0
    if c.lsec < 1.3:
        frame = smoke_wipe(frame, c.lsec / 1.3, SMOKE_COL, reverse=True)
    caption(frame, c, 1.0, [("离开地面", "big"), ("人类第一次乘热气球飞上天空", "mid"), ("巴黎 · 飞行约 25 分钟", "small")])
    return post(frame, c.fi)


# ---------------- 1853 凯利滑翔机（代码绘制） ----------------

def _cayley():
    S = 2
    im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ink = (44, 30, 18, 255)

    def P(pts):
        return [(x * S, y * S) for x, y in pts]
    # 主翼（略带上反的大翼面）与翼肋
    wing = [(560, 430), (1350, 380), (1420, 410), (640, 470)]
    d.polygon(P(wing), fill=(214, 198, 166, 255), outline=ink)
    for k in range(1, 12):
        u = k / 12
        a = (lerp(560, 1350, u), lerp(430, 380, u))
        b = (lerp(640, 1420, u), lerp(470, 410, u))
        d.line(P([a, b]), fill=ink, width=2 * S)
    # 船形吊舱、轮子、支柱
    d.polygon(P([(820, 560), (1140, 560), (1100, 640), (870, 640)]), fill=(150, 120, 84, 255), outline=ink)
    for x0 in range(840, 1120, 12):
        d.line(P([(x0, 565), (x0 + 30, 635)]), fill=(44, 30, 18, 120), width=S)
    for wx in (880, 1080):
        d.ellipse(P([(wx - 26, 640), (wx + 26, 692)]), outline=ink, width=3 * S)
        d.line(P([(wx, 640), (wx, 692)]), fill=ink, width=S)
        d.line(P([(wx - 26, 666), (wx + 26, 666)]), fill=ink, width=S)
    for x0, x1 in ((860, 760), (980, 1000), (1100, 1260)):
        d.line(P([(x0, 560), (x1, 440 - (x1 - 560) * 0.06)]), fill=ink, width=3 * S)
    # 尾翼
    d.line(P([(1140, 590), (1500, 560)]), fill=ink, width=3 * S)
    d.polygon(P([(1460, 520), (1560, 512), (1580, 600), (1480, 606)]), fill=(214, 198, 166, 255), outline=ink)
    d.line(P([(1520, 516), (1530, 603)]), fill=ink, width=2 * S)
    # 乘员
    d.ellipse(P([(955, 505), (985, 535)]), fill=ink)
    d.polygon(P([(950, 535), (990, 535), (1000, 565), (940, 565)]), fill=ink)
    # 地面
    d.line(P([(300, 820), (1700, 800)]), fill=ink, width=2 * S)
    for x0 in range(300, 1700, 16):
        d.line(P([(x0, 820 - (x0 - 300) * 0.014), (x0 - 20, 845 - (x0 - 300) * 0.014)]), fill=(44, 30, 18, 110), width=S)
    im = im.resize((W, H), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    return PAPER * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]


CAYLEY = None


def s_cayley(c):
    global CAYLEY
    if CAYLEY is None:
        CAYLEY = _cayley()
    f = ken(c, CAYLEY, (0.52, 0.52, 1.0), (0.56, 0.47, 1.12))
    f = whip(f, 1 - ease_out_expo(c.since(0) * FPS / 7), 1)
    caption(f, c, 0.4, [("凯利", "big"), ("没有发动机，靠机翼飞起来", "mid"), ("载人滑翔机", "small")], color=INK, scrim=0.0)
    return post(f, c.fi)


# ---------------- 1865 凡尔纳 ----------------

def s_verne(c):
    L = c.lb
    if L < 2:
        f = ken(c, src("verne02"), (0.5, 0.55, 1.0), (0.5, 0.5, 1.12), t0=0, t1=2)
        f = whip(f, 1 - ease_out_expo(c.since(0) * FPS / 7), 1)
    elif L < 4:
        f = ken(c, src("verne01"), (0.5, 0.75, 1.15), (0.52, 0.35, 1.25), t0=2, t1=4)
    elif L < 6:
        f = ken(c, src("verne03"), (0.35, 0.45, 1.1, 0), (0.30, 0.30, 1.25, 3), t0=4, t1=6)
    else:
        f = ken(c, src("verne04"), (0.5, 0.55, 1.05), (0.5, 0.5, 1.2), t0=6, t1=12)
        mix = ease_io(clamp01((L - 8.0) / 1.4))
        if mix > 0:
            cap = ken(c, src("capsule"), (0.5, 0.45, 1.25), (0.5, 0.42, 1.12), t0=8, t1=12)
            f = f * (1 - mix) + cap * mix
    caption(f, c, 0.4, [("凡尔纳", "big"), ("三个人，从佛罗里达出发，最后落进大海", "mid"), ("《从地球到月球》", "small")],
            out=7.8)
    caption(f, c, 8.8, [("一百年后", "big"), ("阿波罗几乎照着写", "mid")])
    sh, e = shake_at(c, 2.0, 12, 0.5)
    sh2, e2 = shake_at(c, 6.0, 10, 0.5)
    return post(f, c.fi, flash=flash_at(c, 2.0, 2, 0.6), shake=(sh[0] + sh2[0], sh[1] + sh2[1]), chroma=4 * (e + e2))


# ---------------- 李林塔尔 ----------------

def s_lilienthal(c):
    if c.lb < 3:
        f = ken(c, src("lil1"), (0.55, 0.48, 1.12, 1.5), (0.62, 0.40, 1.21, -0.5), t0=0, t1=3)
        f = whip(f, 1 - ease_out_expo(c.since(0) * FPS / 7), 1)
    else:
        f = ken(c, src("lil2"), (0.5, 0.5, 1.05), (0.55, 0.45, 1.18), t0=3, t1=6)
        f = whip(f, 1 - ease_out_expo(c.since(3) * FPS / 6), -1)
    caption(f, c, 0.5, [("李林塔尔", "big"), ("近两千次滑翔", "mid"), ("最后一次，他没能平安落地", "small")])
    return post(f, c.fi)


# ---------------- 1903 分屏 ----------------

def _formula():
    parts = [("Δv = v", 0), ("e", 1), ("  ln ( m", 0), ("0", 1), (" / m", 0), ("1", 1), (" )", 0)]
    big, sub = font("serif", 66), font("serif", 38)
    wtot = sum((big if k == 0 else sub).getlength(tx) for tx, k in parts) + 40
    im = Image.new("L", (int(wtot), 130), 0)
    d = ImageDraw.Draw(im)
    x = 20
    for tx, k in parts:
        f = big if k == 0 else sub
        d.text((x, 20 if k == 0 else 62), tx, font=f, fill=255)
        x += f.getlength(tx)
    a = np.asarray(im, np.float32)[..., None] / 255
    return np.ones(a.shape[:2] + (3,), np.float32) * np.array(INK, np.float32), a


FORMULA = _formula()


def s_1903(c):
    s_ = src("wright")
    h, w = s_.shape[:2]
    base = 1280 / max(h, w)
    pc = cover_scale(s_, W // 2, H)
    crops = [(0, 0.57, 0.37, 2.6 * base), (2, 0.86, 0.44, 3.0 * base), (4, 0.52, 0.45, pc * 1.08)]
    seg = [cr for cr in crops if c.lb >= cr[0]][-1]
    nxt = {0: 2, 2: 4, 4: 8}[seg[0]]
    k = (c.lb - seg[0]) / (nxt - seg[0])
    rgb, a = place(s_, seg[1] * w + lerp(-24, 24, k), seg[2] * h, seg[3] * lerp(1.0, 1.06, k), 0, W / 4, H / 2, W // 2, H)
    left = PAPER_DARK[:, : W // 2] * (1 - a) + rgb * a
    fl = flash_at(c, seg[0], 1, 0.5) if seg[0] > 0 else 0
    left = left * (1 - fl) + fl
    left *= 1 - np.clip(1 - yy[:, : W // 2, None] / (H * 0.3), 0, 1) * 0.6
    left *= 1 - np.clip((yy[:, : W // 2, None] - H * 0.6) / (H * 0.4), 0, 1) * 0.7
    right = PAPER[:, W // 2:].copy() * 0.95
    frame = split(left, right, line=ease_out_expo(c.since(0) * FPS / 8))
    rv = clamp01((c.lb - 0.5) / 4.5)
    draw_text(frame, "齐奥尔科夫斯基", "serif", 36, INK, W * 0.75, H * 0.33, clamp01((c.lb - 0.3) * 2), anchor="c",
              shadow=False, tracking=4)
    draw_img(frame, *FORMULA, W * 0.75, H * 0.47, 1.0, rv)
    draw_text(frame, "写下了飞出地球的公式", "serif", 34, INK, W * 0.75, H * 0.60, clamp01((c.lb - 1.0) * 2), anchor="c",
              shadow=False)
    if c.lb > 5:
        draw_text(frame, "“地球是人类的摇篮，但人不能永远生活在摇篮里”", "serif", 26, INK, W * 0.75, H * 0.71,
                  clamp01((c.lb - 5) * 2), anchor="c", shadow=False)
    fr = c.since(0) * FPS
    draw_text(frame, "1903，同一年", "brush", 110, CREAM, W * 0.25, 130, min(fr / 4, 1), lerp(1.25, 1, ease_out_expo(fr / 9)),
              anchor="c")
    draw_text(frame, "莱特兄弟 · 人类第一次驾驶飞机，飞了 12 秒", "serif", 32, CREAM, W * 0.25, H * 0.73,
              clamp01((c.lb - 1.0) * 2), anchor="c")
    sh, _ = shake_at(c, 0, 8, 0.3)
    return post(frame, c.fi, shake=sh)


# ---------------- 1920 纽约时报 ----------------

NEWS, NEWS_HEAD = newspaper("NEW YORK, TUESDAY, JANUARY 13, 1920", ["A SEVERE STRAIN", "ON CREDULITY"])


def s_nyt(c):
    fr = c.since(0) * FPS
    k1 = ease_out_expo(fr / 9)
    scale, rot = lerp(1.1, 0.52, k1), lerp(-14, -4, k1)
    cx, cy, sx = NEWS.shape[1] / 2, NEWS.shape[0] / 2, W * 0.62
    if c.lb >= 2:
        k2 = ease_out_expo(c.since(2) * FPS / 7)
        scale = lerp(0.52, 1.05, k2) * lerp(1, 1.04, (c.lb - 2) / 2)
        rot = lerp(-4, -1.5, k2)
        cx, cy = lerp(cx, NEWS_HEAD[0], k2), lerp(cy, NEWS_HEAD[1], k2)
        sx = lerp(W * 0.62, W * 0.74, k2)
    rgb, a = place(NEWS, cx, cy, scale, rot, sx, H / 2 + 20)
    sh_a = cv2.GaussianBlur(a[..., 0], (0, 0), 18)
    frame = PAPER_DARK * (1 - np.roll(np.roll(sh_a, 14, 0), 10, 1)[..., None] * 0.7)
    frame = frame * (1 - a) + rgb * a
    caption(frame, c, 0.3, [("嘲笑", "big"), ("《纽约时报》社论：", "mid"), ("火箭不可能在真空中飞行", "mid"),
                            ("被嘲笑的人，叫戈达德", "small")], scrim=0.8)
    sh, e = shake_at(c, 2 if c.lb >= 2 else 0, 14, 0.35)
    return post(frame, c.fi, flash=flash_at(c, 2, 1, 0.6), shake=sh, chroma=4 * e)


# ---------------- 1926 戈达德 ----------------

def s_goddard(c):
    f = ken(c, src("goddard"), (0.5, 0.66, 1.0), (0.5, 0.27, 1.05))
    caption(f, c, 0.3, [("戈达德", "big"), ("第一枚液体燃料火箭", "mid"), ("飞了 2.5 秒 · 高约 12.5 米", "small")], scrim=0.8)
    sh, _ = shake_at(c, 0, 10, 0.3)
    return post(f, c.fi, flash=flash_at(c, 0, 2, 0.8), shake=sh)


# ---------------- 1931 皮卡尔（代码绘制） ----------------

def _piccard_layers():
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top, mid, low = np.array([0.03, 0.03, 0.04]), np.array([0.12, 0.10, 0.09]), np.array([0.62, 0.55, 0.46])
    sky = np.where(y < 0.6, top + (mid - top) * (y / 0.6), mid + (low - mid) * np.clip((y - 0.6) / 0.4, 0, 1))
    sky = np.broadcast_to(sky, (H, W, 3)).copy()
    d = np.sqrt((xx - W / 2) ** 2 + (yy - (H + 2600)) ** 2)
    earth = np.clip((2600 + 230 - d) / 3, 0, 1)[..., None]
    haze = np.exp(-((d - 2830) / 60) ** 2)[..., None]
    sky = sky * (1 - earth) + (np.array([0.42, 0.37, 0.30]) * (0.85 + 0.2 * fbm(H, W, 5, 61)[..., None])) * earth
    sky = sky + haze * np.array([0.5, 0.45, 0.36]) * 0.6
    S = 2
    im = Image.new("RGBA", (900 * S, 1100 * S), (0, 0, 0, 0))
    dd = ImageDraw.Draw(im)
    cx = 450
    env_pts = []
    for t_ in np.linspace(0, np.pi, 60):
        env_pts.append((cx + np.sin(t_) * 230 * (0.75 + 0.25 * np.sin(t_)), 60 + (1 - np.cos(t_)) * 230))
    env_pts += [(cx + 40, 560), (cx - 40, 560)]
    env_pts = env_pts[:60] + [(cx - x + cx, y_) for x, y_ in reversed(env_pts[:60])]
    dd.polygon([(x * S, y_ * S) for x, y_ in env_pts], fill=(218, 206, 182, 255), outline=(60, 48, 36, 255))
    for k in range(-4, 5):
        dd.line([((cx + k * 40) * S, 120 * S), ((cx + k * 8) * S, 560 * S)], fill=(150, 136, 112, 255), width=2 * S)
    for k in (-1, 1):
        dd.line([((cx + k * 30) * S, 560 * S), ((cx + k * 70) * S, 760 * S)], fill=(60, 48, 36, 255), width=2 * S)
    dd.ellipse([(cx - 95) * S, 740 * S, (cx + 95) * S, 930 * S], fill=(196, 196, 190, 255), outline=(60, 60, 58, 255), width=3 * S)
    dd.ellipse([(cx - 60) * S, 780 * S, (cx - 20) * S, 820 * S], fill=(90, 90, 88, 255))
    dd.ellipse([(cx + 10) * S, 760 * S, (cx + 50) * S, 800 * S], fill=(90, 90, 88, 255))
    dd.ellipse([(cx - 70) * S, 760 * S, (cx + 10) * S, 800 * S], fill=(240, 240, 232, 120))
    im = im.resize((900, 1100), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    return sky.astype(np.float32), a


PICCARD = None


def s_piccard(c):
    global PICCARD
    if PICCARD is None:
        PICCARD = _piccard_layers()
    sky, bal = PICCARD
    f = sky.copy()
    y0 = lerp(320, -120, ease_io(c.k))
    rgb, a = bal[..., :3], bal[..., 3:]
    x0 = 900
    h, w = a.shape[:2]
    ys, ye = int(max(y0, 0)), int(min(y0 + h, H))
    if ye > ys:
        sub_a = a[ys - int(y0):ye - int(y0)]
        sub_r = rgb[ys - int(y0):ye - int(y0)]
        f[ys:ye, x0:x0 + w] = f[ys:ye, x0:x0 + w] * (1 - sub_a) + sub_r * sub_a
    f = duotone(f, SEPIA_D, SEPIA_L, 1.0, 1.1)
    caption(f, c, 0.3, [("皮卡尔", "big"), ("第一次乘气球进入平流层", "mid"), ("15.8 公里 · 密封舱", "small")])
    return post(f, c.fi)


# ---------------- 1946 V-2 ----------------

def s_v2(c):
    s_ = src("v2")
    emerge = ease_io(clamp01(c.lb / 2.0))
    f = ken(c, s_, (0.5, 0.5, 1.04, -3), (0.5, 0.5, 1.12, 0)) * emerge
    f = np.clip(f + cv2.resize(np.random.default_rng(c.fi).standard_normal((H // 2, W // 2)).astype(np.float32),
                               (W, H))[..., None] * 0.18 * (1 - emerge * 0.7), 0, 1)
    caption(f, c, 0.8, [("从太空看见地球", "big"), ("人类第一次", "mid"), ("拍下它的，是一枚战争武器 V-2 · 约 105 公里高空", "small")])
    return post(f, c.fi, grain=0.06)


# ---------------- 1947 X-1 音爆 ----------------

def s_x1(c):
    s_ = src("x1")
    h, w = s_.shape[:2]
    sc = cover_scale(s_)
    if c.lb < 2:
        k = ease_in(c.lb / 2, 2.6)
        rgb, a = place(s_, lerp(0.50, 0.22, k) * w, lerp(0.48, 0.44, k) * h, sc * lerp(1.05, 1.75, k))
        f = PAPER_DARK * (1 - a) + rgb * a
        f = motion_blur(f, 6 + 90 * k, 0)
        lines = np.zeros((H, W), np.float32)
        r = np.random.default_rng(c.fi)
        for _ in range(int(10 + 50 * k)):
            y = int(r.uniform(0, H))
            x0 = int(r.uniform(-200, W))
            cv2.line(lines, (x0, y), (x0 + int(r.uniform(200, 700)), y), 1.0, 1)
        f = f + cv2.GaussianBlur(lines, (0, 0), 0.8)[..., None] * 0.5 * k
        return post(f, c.fi)
    k = (c.lb - 2) / 2
    rgb, a = place(s_, 0.52 * w, 0.5 * h, sc * lerp(1.0, 1.04, k), lerp(0.8, 0, k))
    f = PAPER_DARK * (1 - a) + rgb * a
    caption(f, c, 2.15, [("1.06 马赫", "big"), ("突破音障", "mid"), ("耶格尔 · 贝尔 X-1", "small")], y=150)
    sh, e = shake_at(c, 2, 34, 0.9)
    return post(f, c.fi, flash=flash_at(c, 2, 3), shake=sh, chroma=9 * e)


def s_silence(c):
    return post(np.zeros((H, W, 3), np.float32), c.fi, grain=0.02, vignette=False, weave=False)


SHOTS = [Shot(8, s_card, name="3-01"), Shot(12, s_wanhu, name="3-02"), Shot(8, s_balloon, name="3-03"),
         Shot(4, s_cayley, name="3-04"), Shot(12, s_verne, name="3-05"), Shot(6, s_lilienthal, name="3-06"),
         Shot(8, s_1903, name="3-07"), Shot(4, s_nyt, name="3-08"), Shot(4, s_goddard, name="3-09"),
         Shot(4, s_piccard, name="3-10"), Shot(4, s_v2, name="3-11"), Shot(4, s_x1, name="3-12"),
         Shot(1, s_silence, name="3-13")]
_ST = np.concatenate([[0], np.cumsum([s.beats for s in SHOTS])])

TICKER = YearTicker([(_ST[1], "明朝", "传说"), (_ST[2], 1783, "11.21"), (_ST[3], 1853, ""), (_ST[4], 1865, ""),
                     (_ST[5], 1891, "—1896"), (_ST[6], 1903, "12.17"), (_ST[7], 1920, "01.13"),
                     (_ST[8], 1926, "03.16"), (_ST[9], 1931, "05.27"), (_ST[10], 1946, "10.24"),
                     (_ST[11], 1947, "10.14")])


def overlay(img, c):
    b = c.b
    if b < _ST[1] or b >= _ST[12]:
        return img
    a = clamp01((b - _ST[1]) * 2) * (1 - clamp01((b - (_ST[12] - 0.3)) * 4))
    return TICKER.draw(img, b, a)


# ---------------- 配乐 ----------------

ROOTS = {"Dm": 38, "Bb": 34, "F": 41, "C": 36, "Gm": 43, "A": 33}
TRIADS = {"Dm": [50, 53, 57], "Bb": [46, 50, 53], "F": [45, 48, 53], "C": [48, 52, 55], "Gm": [43, 50, 55],
          "A": [45, 49, 52]}
QUALITY = {"Dm": 3, "Bb": 4, "F": 4, "C": 4, "Gm": 3, "A": 4}
BARS = ["Dm", "Dm", "Dm", "Dm", "A", "Dm", "Bb", "F", "C", "Dm", "Bb", "Gm", "A", "Dm", "Bb", "Gm", "A", "F", "Dm",
        "A", "A"]


def music(sc, ch):
    T, t = ch.T, ch.t

    def ost(b0, b1, gain=0.35, chord=None):
        b = b0
        while b < b1 - 1e-6:
            cn = chord or BARS[int(b // 4)]
            r = ROOTS[cn]
            pat = [0, 0, 7, 0, 12, 0, 7, 10 if QUALITY[cn] == 3 else 9]
            m = r + pat[int(round((b % 4) * 2)) % 8]
            d = (t(b + 0.5) - t(b)) * 0.8
            sc.add("bgm", M.strings_short(m, d), T(b), gain * (1.15 if b % 1 == 0 else 0.85), pan=-0.25)
            sc.add("bgm", M.strings_short(m + 12, d), T(b), gain * 0.35, pan=0.3)
            b += 0.5

    def pads(b0, b1, gain=0.22):
        b = b0
        while b < b1 - 1e-6:
            nxt = min((int(b // 4) + 1) * 4, b1)
            sc.add("bgm", M.pad(TRIADS[BARS[int(b // 4)]], t(nxt) - t(b) + 0.4), T(b), gain)
            b = nxt

    def timp(b0, b1, every=2.0, gain=0.7):
        b = b0
        while b < b1 - 1e-6:
            cn = BARS[int(b // 4)]
            m = ROOTS[cn] if b % 4 == 0 else ROOTS[cn] + 7 - 12
            sc.add("bgm", M.timpani(m, vel=1.0 if b % 4 == 0 else 0.7), T(b), gain)
            b += every

    def stab(b, gain=0.5):
        cn = BARS[int(b // 4)]
        sc.add("bgm", M.brass_stab([m + 12 for m in TRIADS[cn]] + [ROOTS[cn] + 12]), T(b), gain)

    S_ = _ST
    # 章节卡
    sc.add("bgm", M.pad([38, 45, 50], t(8) + 0.2), T(0), 0.18)
    b = 4.0
    while b < 7.75:
        sc.add("bgm", M.timpani(38, 0.6, 0.25 + 0.75 * (b - 4) / 4), T(b), 0.6)
        b += 0.25
    sc.add("sfx", M.stamp(), T(2.5), 0.7)
    sc.add("sfx", M.real("FX/FX-11/FX-11_01_short_whoosh_pixabay.mp3"), T(7.5), 0.4)
    # 万户
    w0 = S_[1]
    ost(w0, w0 + 7.5)
    pads(w0, w0 + 7.5, 0.16)
    timp(w0, w0 + 7.5)
    stab(w0, 0.45)
    sc.add("sfx", M.real("FX/FX-08/FX-08_01_fuse_burning_pixabay.mp3", t(w0 + 5) - t(w0)), T(w0), 0.3, pan=0.2)
    sc.gate(T(w0 + 7.5), T(w0 + 8), 0.08)
    t0, t1 = T(w0 + 8), T(w0 + 12)
    roar = M.real_loudest("FX/FX-02/FX-02_01_falcon9_pad_mics_nasa.wav", t1 - t0 + 1.0)
    x = np.arange(len(roar)) / M.SR
    roar = M.filt(roar, "low", 5000) * np.clip(x / 0.08, 0, 1) * (0.75 + 0.25 * np.clip(x / 1.5, 0, 1)) \
        * np.clip((t1 - t0 + 0.6 - x) / 0.8, 0, 1)
    sc.add("sfx", roar, t0, 0.6)
    sc.add("sfx", M.boom(1.6, 48, 30, 0.1), t0, 0.3)
    sc.add("bgm", M.timpani(33, vel=1.2), t0, 0.8)
    sc.add("bgm", M.brass_stab([69, 73, 76, 81, 57], 0.9), t0, 0.55)
    sc.add("bgm", M.cymbal(2.0), t0, 0.2)
    ost(w0 + 8, w0 + 12, 0.38, chord="A")
    sc.add("bgm", M.pad([45, 52, 57, 61, 64], t1 - t0 + 0.3), t0, 0.45)
    for k, m in enumerate([69, 71, 73, 76]):
        rb = w0 + 9 + k * 0.75
        sc.add("bgm", M.horn(m, t(rb + 0.75) - t(rb)), T(rb), 0.35)
    b = w0 + 10.5
    while b < w0 + 11.9:
        sc.add("bgm", M.snare(0.3 + 0.7 * (b - w0 - 10.5) / 1.5), T(b), 0.32)
        b += 0.25 if b < w0 + 11.25 else 0.125
    sc.add("sfx", M.real("FX/FX-11/FX-11_01_short_whoosh_pixabay.mp3"), T(w0 + 11.2), 0.5)
    # 热气球起：全编制
    b_ = S_[2]
    stab(b_, 0.6)
    sc.add("bgm", M.timpani(38, vel=1.2), T(b_), 0.8)
    sc.add("bgm", M.cymbal(), T(b_), 0.22)
    ost(b_, S_[10])
    pads(b_, S_[10])
    timp(b_ + 2, S_[10])
    sc.add("sfx", M.real("FX/FX-07/FX-07_03_fire_sounds_pixabay.mp3", t(b_ + 5) - t(b_ + 0.5), 0.4), T(b_ + 0.5), 0.2)
    melody = [(0, 62, 2), (2, 65, 1), (3, 69, 1), (4, 67, 2), (6, 65, 1), (7, 64, 1), (8, 69, 2), (10, 72, 1),
              (11, 74, 1), (12, 72, 2), (14, 69, 2), (16, 67, 2), (18, 70, 1), (19, 69, 1), (20, 69, 4),
              (26, 74, 2), (28, 72, 1), (29, 73, 1), (30, 74, 4), (36, 77, 2), (38, 76, 2), (40, 74, 4)]
    for o, m, d in melody:
        sc.add("bgm", M.horn(m, t(b_ + o + d) - t(b_ + o)), T(b_ + o), 0.2, pan=0.15)
    # 凯利、凡尔纳、李林塔尔
    sc.add("sfx", M.real("FX/FX-07/FX-07_01_soft_wind_pixabay.mp3", t(S_[4]) - t(S_[3]), 0.2), T(S_[3]), 0.3)
    sc.add("sfx", M.real("FX/FX-11/FX-11_01_short_whoosh_pixabay.mp3"), T(S_[3]) - 0.2, 0.35)
    sc.add("sfx", M.real("FX/FX-04/FX-04_01_projector_16mm_pixabay.mp3", t(S_[5]) - t(S_[4]), 0.2), T(S_[4]), 0.2)
    sc.add("sfx", M.boom(1.8, 60, 30, 0.15), T(S_[4] + 2), 0.35)
    sc.add("sfx", M.whoosh(0.8, 300, 3000), T(S_[4] + 4) - 0.2, 0.25)
    sc.add("sfx", M.boom(1.4, 70, 40, 0.1), T(S_[4] + 6), 0.25)
    sc.add("bgm", M.bell(81, 3.0), T(S_[4] + 8.2), 0.25)
    sc.add("sfx", M.real("FX/FX-11/FX-11_01_short_whoosh_pixabay.mp3"), T(S_[5]) - 0.25, 0.45)
    sc.add("sfx", M.real("FX/FX-07/FX-07_02_harsh_wind_pixabay.mp3", t(S_[6] - 0.5) - t(S_[5]), 0.2), T(S_[5]), 0.3)
    # 1903
    stab(S_[6], 0.6)
    sc.add("bgm", M.cymbal(1.5), T(S_[6]), 0.15)
    sc.add("sfx", M.real("FX/FX-14/FX-14_01_small_propeller_plane_freesound.wav", t(S_[6] + 6) - t(S_[6]), 0.3),
           T(S_[6]), 0.3, pan=-0.5)
    for o in (2, 4):
        sc.add("sfx", M.whoosh(0.18, 800, 6000), T(S_[6] + o) - 0.06, 0.2)
    # 纽约时报
    sc.add("sfx", M.real("FX/FX-13/FX-13_02_printing_press_rhythm_freesound.flac", t(S_[8]) - t(S_[7]), 0.1), T(S_[7]), 0.4)
    stab(S_[7], 0.6)
    stab(S_[7] + 2, 0.5)
    sc.add("sfx", M.real("FX/FX-11/FX-11_01_short_whoosh_pixabay.mp3"), T(S_[7] + 2) - 0.15, 0.4)
    # 戈达德
    timp(S_[8], S_[9], every=0.5, gain=0.4)
    sc.add("sfx", M.filt(M.noise(2.5), "band", [400, 5000]) * np.clip(np.arange(int(2.5 * M.SR)) / M.SR / 0.3, 0, 1),
           T(S_[8] + 0.5), 0.12)
    sc.add("bgm", M.cymbal(1.2), T(S_[8]), 0.2)
    # 皮卡尔：高空风
    sc.add("sfx", M.real("FX/FX-07/FX-07_01_soft_wind_pixabay.mp3", t(S_[10]) - t(S_[9]), 0.3, start=4), T(S_[9]), 0.3)
    # V-2：鼓与固定音型撤掉，只留铺底和短波
    sc.add("sfx", M.real("FX/FX-09/FX-09_02_shortwave_tuning_noise_pixabay.mp3", t(S_[11] + 1) - t(S_[10]), 0.4),
           T(S_[10]), 0.25)
    sc.add("bgm", M.pad([50, 57, 62], t(S_[11] + 2) - t(S_[10])), T(S_[10]), 0.2)
    # X-1：上升音 + 小军鼓滚奏 → 音爆（真实音爆录音 + 低频，限幅器保护，不刺耳）
    x0 = S_[11]
    sc.add("sfx", M.riser(t(x0 + 1.85) - t(x0 - 2)), T(x0 - 2), 0.4)
    b = x0
    while b < x0 + 1.85:
        sc.add("bgm", M.snare(0.3 + 0.7 * (b - x0) / 2), T(b), 0.32)
        b += 0.25 if b < x0 + 1 else 0.125
    sb = M.real("FX/FX-15/FX-15_02_fighter_real_sonic_boom_freesound.flac", 3.0, 0.005)
    sc.add("sfx", M.filt(sb, "low", 6000) if sb is not None else None, T(x0 + 2), 0.55)
    sc.add("sfx", M.boom(2.5, 55, 24, 0.2), T(x0 + 2), 0.5)
    sc.add("bgm", M.timpani(38, vel=1.3), T(x0 + 2), 0.9)
    sc.add("bgm", M.cymbal(2.0), T(x0 + 2), 0.3)
    stab(x0 + 2, 0.7)
    sc.add("sfx", M.real("FX/FX-11/FX-11_02_cinematic_low_hit_pixabay.mp3"), T(x0 + 2), 0.6)
    # 动态：V-2 收到 −8 dB，X-1 滚奏期间推回去，音爆满电平落下（自动电平之后施加，不会被抬平）
    sc.level(T(S_[10]), T(x0), -8)
    sc.level(T(x0), T(x0 + 1.95), -8, -3)
    stab(x0 + 2, 0.65)
    sc.silence(T(S_[12]), T(S_[13]))
    # 年份滚动时的电传声
    tty = M.real("FX/FX-12/FX-12_01_vintage_teletype_freesound.wav", 0.35, 0.02)
    for k in range(2, 12):
        sc.add("sfx", tty, T(S_[k]), 0.18)


CHAPTER = Chapter("pioneers", "先驱", 96, SHOTS, bpm1=112, ramp_beats=44, overlay=overlay, music=music)
