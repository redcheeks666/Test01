"""先驱章节奏样片：逐帧渲染画面，与 out/music.wav 合成为 out/ch3_demo.mp4。

素材暂用 assets_review/previews 里的预览图（长边 1280），原件入库后把 SRC 指向原件即可。
"""
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from timeline import DURATION, FPS, WANHU_END, SHIFT, t, t_real, beat_at

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PREV = ROOT / "assets_review" / "previews"
FONTS = ROOT / "edit" / "fonts"
OUT = HERE / "out"
W, H = 1920, 1080
rng = np.random.default_rng(3)

INK = (0.16, 0.11, 0.07)
CREAM = (0.99, 0.96, 0.89)
DUO_DARK = np.array([0.09, 0.06, 0.04])
DUO_LIGHT = np.array([0.94, 0.88, 0.76])
SEAL_RED = (0.72, 0.13, 0.10)


# ---------------- 基础工具 ----------------

def ease_io(x):
    x = np.clip(x, 0, 1)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_expo(x):
    x = np.clip(x, 0, 1)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_in(x, p=2.5):
    return np.clip(x, 0, 1) ** p


def lerp(a, b, k):
    return a + (b - a) * k


def load(name):
    p = next(PREV.glob(name + "*"))
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def duotone(img, gamma=1.0, contrast=1.15):
    g = img @ np.array([0.3, 0.59, 0.11], dtype=np.float32)
    g = np.clip((g - 0.5) * contrast + 0.5, 0, 1) ** gamma
    return DUO_DARK + g[..., None] * (DUO_LIGHT - DUO_DARK)


def fbm(h, w, octaves=5, seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = 2 ** (o + 2)
        small = r.random((s * h // w + 2, s + 2)).astype(np.float32)
        out += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp
        tot += amp
        amp *= 0.5
    return out / tot


def make_paper(dark=False):
    n = fbm(H, W, 6, seed=11 if dark else 5)
    fib = cv2.GaussianBlur((rng.random((H, W)) > 0.996).astype(np.float32), (0, 0), 0.8)
    fib = cv2.filter2D(fib, -1, np.ones((1, 9), np.float32) / 9)
    if dark:
        base = np.array([0.11, 0.085, 0.065], np.float32)
        p = base * (0.8 + 0.4 * n[..., None]) + fib[..., None] * 0.05
    else:
        base = np.array([0.91, 0.86, 0.75], np.float32)
        p = base * (0.88 + 0.16 * n[..., None]) - fib[..., None] * 0.15
    return np.clip(p, 0, 1)


PAPER = make_paper()
PAPER_DARK = make_paper(dark=True)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = np.clip(1 - 0.55 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2), 0, 1)[..., None]


def place(src, cx, cy, scale, rot=0.0, sx=W / 2, sy=H / 2, out_w=W, out_h=H):
    """把 src 上的点 (cx, cy)（像素）放到屏幕 (sx, sy)，缩放 scale、旋转 rot 度。返回 (rgb, alpha)。"""
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), rot, scale)
    m[0, 2] += sx - cx
    m[1, 2] += sy - cy
    rgb = cv2.warpAffine(src, m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    a = cv2.warpAffine(np.ones(src.shape[:2], np.float32), m, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return rgb, a[..., None]


def comp(bg, rgb, a, shadow=0.0):
    if shadow:
        sh = cv2.GaussianBlur(a[..., 0], (0, 0), 18)
        sh = np.roll(np.roll(sh, 14, 0), 10, 1)[..., None]
        bg = bg * (1 - sh * shadow)
    return bg * (1 - a) + rgb * a


def cover_scale(src, w=W, h=H):
    return max(w / src.shape[1], h / src.shape[0])


def motion_blur(img, length, angle=0.0):
    length = int(length)
    if length < 2:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0), (length, length))
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def sharpen(img, amt=0.6):
    blur = cv2.GaussianBlur(img, (0, 0), 1.6)
    return np.clip(img + (img - blur) * amt, 0, 1)


# ---------------- 文字 ----------------

@lru_cache(None)
def font(kind, size):
    path = {"brush": "MaShanZheng-Regular.ttf", "serif": "NotoSerifSC[wght].ttf",
            "mono": "IBMPlexMono-Regular.ttf", "wild": "ZhiMangXing-Regular.ttf"}[kind]
    f = ImageFont.truetype(str(FONTS / path), size)
    if kind == "serif":
        try:
            f.set_variation_by_axes([600])
        except Exception:
            pass
    return f


@lru_cache(None)
def text_img(text, kind, size, color, tracking=0):
    f = font(kind, size)
    pad = size // 2
    widths = [f.getbbox(ch)[2] - f.getbbox(ch)[0] if ch != " " else size // 3 for ch in text]
    tw = sum(widths) + tracking * (len(text) - 1) + 2 * pad
    th = int(size * 1.45) + 2 * pad
    im = Image.new("L", (tw, th), 0)
    d = ImageDraw.Draw(im)
    x = pad
    for ch, w in zip(text, widths):
        d.text((x - f.getbbox(ch)[0], pad), ch, font=f, fill=255)
        x += w + tracking
    a = np.asarray(im, np.float32)[..., None] / 255.0
    rgb = np.ones((th, tw, 3), np.float32) * np.array(color, np.float32)
    return rgb, a, pad


def draw_text(frame, text, kind, size, color, x, y, alpha=1.0, scale=1.0, tracking=0,
              shadow=True, reveal=1.0, anchor="lt"):
    """在 frame 上画文字。x, y 为文字左上角（anchor='lt'）或中心（anchor='c'）。reveal 控制从左到右显现。"""
    if alpha <= 0.001:
        return frame
    rgb, a, pad = text_img(text, kind, size, tuple(color), tracking)
    if reveal < 1:
        a = a.copy()
        edge = int(a.shape[1] * reveal)
        a[:, edge:] = 0
    if scale != 1.0:
        nw, nh = max(1, int(a.shape[1] * scale)), max(1, int(a.shape[0] * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)[..., None]
    h, w = a.shape[:2]
    if anchor == "c":
        x0, y0 = int(x - w / 2), int(y - h / 2)
    else:
        x0, y0 = int(x - pad * scale), int(y - pad * scale)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xb <= xa or yb <= ya:
        return frame
    sa = a[ya - y0:yb - y0, xa - x0:xb - x0] * alpha
    sr = rgb[ya - y0:yb - y0, xa - x0:xb - x0]
    region = frame[ya:yb, xa:xb]
    if shadow:
        dark = sum(color) > 1.5
        sh = cv2.GaussianBlur(sa[..., 0], (0, 0), 7)[..., None]
        tight = cv2.GaussianBlur(sa[..., 0], (0, 0), 1.8)[..., None]
        if dark:
            region = region * (1 - np.clip(sh * 0.9 + tight * 1.2, 0, 0.92))
        else:
            region = region * (1 - sh * 0.25)
    frame[ya:yb, xa:xb] = region * (1 - sa) + sr * sa
    return frame


GRAD_L = np.clip(1 - xx / (W * 0.55), 0, 1)[..., None] ** 1.6


def caption(frame, b, b_in, lines, color=CREAM, x=110, y=150, b_out=None, scrim=None):
    """三级文字：lines = [(文字, 级别)]，级别 big / mid / small / date。大字在 b_in 冲入，其余依次延后半拍。
    scrim > 0 时在左侧压一层暗渐变，保证文字在复杂背景上可读。"""
    if scrim is None:
        scrim = 0.78 if sum(color) > 1.5 else 0.0
    if scrim and b >= b_in:
        k = min((t(b) - t(b_in)) * FPS / 6, 1)
        frame *= 1 - GRAD_L * scrim * k
    spec = {"big": ("brush", 132, 0), "mid": ("serif", 46, 2), "small": ("serif", 31, 2), "date": ("mono", 32, 3)}
    cy = y
    delay = 0.0
    for text, lvl in lines:
        kind, size, tr = spec[lvl]
        bi = b_in + delay
        k = (b - bi) * 60 / 100  # 约等于秒
        if b >= bi:
            fr = (t(b) - t(bi)) * FPS
            if lvl == "big":
                a = min(fr / 4, 1)
                s = lerp(1.28, 1.0, ease_out_expo(fr / 9))
                draw_text(frame, text, kind, size, color, x, cy, a, s, tr)
            elif lvl == "date":
                n = int(fr)  # 一帧一个字符
                draw_text(frame, text, kind, size, color, x, cy, 1.0, 1.0, tr,
                          reveal=min(1, n / max(len(text), 1)))
            else:
                a = min(fr / 8, 1)
                draw_text(frame, text, kind, size, color, x, cy + (1 - ease_out_expo(fr / 12)) * 22, a, 1.0, tr)
        cy += {"big": 158, "mid": 70, "small": 50, "date": 54}[lvl]
        delay += 0.5 if lvl in ("big", "date") else 0.35
    del k
    return frame


# ---------------- 后期 ----------------

GRAIN_SMALL = [rng.standard_normal((H // 2, W // 2)).astype(np.float32) for _ in range(12)]


def post(frame, fi, flash=0.0, shake=(0, 0), chroma=0.0, grain=0.045, vignette=True, weave=True):
    if weave:
        dx, dy = np.sin(fi * 0.71) * 0.8 + np.sin(fi * 1.93) * 0.5, np.sin(fi * 0.53 + 1) * 0.9
        shake = (shake[0] + dx, shake[1] + dy)
    if abs(shake[0]) > 0.05 or abs(shake[1]) > 0.05:
        m = np.float32([[1.02, 0, shake[0] - W * 0.01], [0, 1.02, shake[1] - H * 0.01]])
        frame = cv2.warpAffine(frame, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if chroma > 0.3:
        c = int(round(chroma))
        frame = frame.copy()
        frame[..., 0] = np.roll(frame[..., 0], c, 1)
        frame[..., 2] = np.roll(frame[..., 2], -c, 1)
    if vignette:
        frame = frame * (0.35 + 0.65 * VIGNETTE)
    if grain:
        g = cv2.resize(GRAIN_SMALL[fi % len(GRAIN_SMALL)], (W, H), interpolation=cv2.INTER_LINEAR)
        frame = frame + g[..., None] * grain
    if flash > 0:
        frame = frame * (1 - flash) + flash
    return np.clip(frame, 0, 1)


def shake_at(b, b_hit, amp, decay_s=0.8):
    dt = t(b) - t(b_hit)
    if dt < 0:
        return (0.0, 0.0), 0.0
    e = np.exp(-dt / decay_s * 4)
    return (amp * e * np.sin(dt * 67), amp * e * np.cos(dt * 53)), e


def flash_at(b, b_hit, frames=3):
    fr = (t(b) - t(b_hit)) * FPS
    if 0 <= fr < frames:
        return 1.0 - fr / frames * 0.4
    if frames <= fr < frames + 5:
        return 0.6 * (1 - (fr - frames) / 5)
    return 0.0


# ---------------- 素材 ----------------

sys.path.insert(0, str(ROOT / "edit"))
from assets import load_image  # noqa: E402

SRC = {k: load_image(v) for k, v in {
    "balloon": "PD-08", "wright": "PD-11", "lilienthal": "PD-10",
    "goddard": "NA-06", "v2": "NA-19", "x1": "NA-20"}.items()}
# 镜头里的绝对缩放按预览图（长边 1280）设计，换原件后用 K 折算
K = {k: 1280 / max(v.shape[:2]) for k, v in SRC.items()}
DUO = {k: sharpen(duotone(v, gamma=1.08, contrast=1.25 if k != "v2" else 1.4)) for k, v in SRC.items()}
DUO["x1"] = DUO["x1"][: int(SRC["x1"].shape[0] * 0.66)]
_lh, _lw = DUO["lilienthal"].shape[:2]
DUO["lilienthal"] = DUO["lilienthal"][int(_lh * 0.08):int(_lh * 0.8), int(_lw * 0.07):int(_lw * 0.93)]


def smoke_layer(seed, h=H, w=W):
    return fbm(h, w, 6, seed)


SMOKE = [smoke_layer(s) for s in (21, 22)]


from wanhu import build as _wanhu_build  # noqa: E402

WANHU = _wanhu_build()


def _along(pts, p):
    """折线 pts 上比例 p 处的点。"""
    seg = [np.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1)]
    dist = p * sum(seg)
    for k, L in enumerate(seg):
        if dist <= L or k == len(seg) - 1:
            q = min(dist / L, 1)
            return lerp(pts[k][0], pts[k + 1][0], q), lerp(pts[k][1], pts[k + 1][1], q)
        dist -= L


def spark(frame, x, y, fi, size=16, strength=1.4):
    g = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * size ** 2)))[..., None]
    frame += g * np.array([1.0, 0.56, 0.18]) * strength * (0.8 + 0.2 * np.sin(fi * 2.7))
    r = np.random.default_rng(fi * 13 + int(x))
    for _ in range(7):
        cv2.circle(frame, (int(x + r.normal(0, size * 1.3)), int(y + r.normal(-size * 0.5, size))), 2, (1.0, 0.82, 0.45), -1)
    return frame


def newspaper():
    """1920 年 1 月 13 日《纽约时报》社论版的重绘（示意，非原版扫描）。"""
    pw, ph = 1500, 1900
    im = Image.new("RGB", (pw, ph), (226, 214, 186))
    d = ImageDraw.Draw(im)
    ink = (34, 26, 20)
    serif = ImageFont.truetype(str(FONTS / "NotoSerifSC[wght].ttf"), 118)
    try:
        serif.set_variation_by_axes([900])
    except Exception:
        pass
    small = ImageFont.truetype(str(FONTS / "IBMPlexMono-Regular.ttf"), 28)
    head = ImageFont.truetype(str(FONTS / "NotoSerifSC[wght].ttf"), 70)
    try:
        head.set_variation_by_axes([800])
    except Exception:
        pass
    d.text((pw / 2, 120), "The New York Times", font=serif, fill=ink, anchor="mm")
    d.line([(60, 210), (pw - 60, 210)], fill=ink, width=4)
    d.text((pw / 2, 240), "NEW YORK, TUESDAY, JANUARY 13, 1920", font=small, fill=ink, anchor="mm")
    d.line([(60, 268), (pw - 60, 268)], fill=ink, width=2)
    r = np.random.default_rng(4)
    cols = 5
    cw = (pw - 120) / cols
    bx0, bx1 = 60 + 2 * cw - cw * 0.5, 60 + 3 * cw + cw * 0.5
    for c in range(cols):
        x0 = 60 + c * cw + 12
        y = 300
        while y < ph - 60:
            if x0 + cw - 24 > bx0 - 10 and x0 < bx1 + 10 and 300 <= y < 820:
                y = 820
                continue
            lw = cw - 24 if r.random() > 0.12 else (cw - 24) * r.uniform(0.3, 0.8)
            d.rectangle([x0, y, x0 + lw, y + 9], fill=(110, 98, 84))
            y += 21 if r.random() > 0.06 else 46
        if c:
            d.line([(60 + c * cw, 290), (60 + c * cw, ph - 60)], fill=ink, width=1)
    # 社论标题框（第 3 栏上方，跨两栏）
    d.rectangle([bx0, 310, bx1, 800], outline=ink, width=3)
    d.text(((bx0 + bx1) / 2, 380), "A SEVERE STRAIN", font=head, fill=ink, anchor="mm")
    d.text(((bx0 + bx1) / 2, 470), "ON CREDULITY", font=head, fill=ink, anchor="mm")
    d.line([(bx0 + 60, 530), (bx1 - 60, 530)], fill=ink, width=2)
    for i in range(10):
        lw = (bx1 - bx0 - 80) * (1 if i < 9 else 0.5)
        d.rectangle([bx0 + 40, 560 + i * 22, bx0 + 40 + lw, 569 + i * 22], fill=(90, 80, 68))
    arr = np.asarray(im, np.float32) / 255.0
    arr = arr * (0.9 + 0.12 * fbm(ph, pw, 5, 9)[..., None])
    return np.clip(arr, 0, 1), ((bx0 + bx1) / 2, 450)


NEWS, NEWS_HEAD = newspaper()


def seal(size=96, text="天问"):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([4, 4, size - 4, size - 4], radius=10, fill=(184, 34, 26, 255))
    f = font("brush", int(size * 0.4))
    for k, ch in enumerate(text):  # 竖排两字
        d.text((size / 2, size * (0.3 + 0.4 * k) + 2), ch, font=f, fill=(245, 228, 210, 0), anchor="mm")
    arr = np.asarray(im, np.float32) / 255.0
    # 印泥斑驳
    mott = (fbm(size, size, 4, 31) > 0.38).astype(np.float32)
    arr[..., 3] *= mott
    return arr[..., :3], arr[..., 3:]


SEAL = seal()


# ---------------- 镜头 ----------------

def chapter_card(b, fi, b0, name, exit_zoom=True):
    """统一章节卡：纸本底纹，两字章节名（毛笔大字），第 6 拍盖印，最后半拍冲进墨迹。"""
    frame = PAPER.copy()
    lb = b - b0
    s_ = lerp(1.06, 1.0, ease_io(min(max((lb - 0.5) / 7, 0), 1)))
    fr = (t(b) - t(b0 + 0.3)) * FPS
    ink_a = np.clip(fr / 10, 0, 1)
    draw_text(frame, name, "brush", int(300 * s_), INK, W / 2, H / 2 - 20, ink_a, 1.0, tracking=30,
              anchor="c", shadow=False)
    if lb >= 2.5:
        fr3 = (t(b) - t(b0 + 2.5)) * FPS
        ss = lerp(1.5, 1.0, ease_out_expo(fr3 / 5))
        rgb, a = SEAL
        sz = int(96 * ss)
        rgb_s, a_s = cv2.resize(rgb, (sz, sz)), cv2.resize(a, (sz, sz))[..., None]
        x0, y0 = int(W / 2 + 330 - sz / 2), int(H / 2 + 110 - sz / 2)
        k = min(fr3 / 2, 1)
        reg = frame[y0:y0 + sz, x0:x0 + sz]
        frame[y0:y0 + sz, x0:x0 + sz] = reg * (1 - a_s * k) + rgb_s * a_s * k
    if exit_zoom and lb > 7.5:
        z = ease_in((lb - 7.5) / 0.5, 2)
        frame, _ = place(frame, W / 2, H / 2 - 20, lerp(1.0, 2.4, z))
        frame = motion_blur(frame, 40 * z, 90)
    sh, _ = shake_at(b, b0 + 2.5, 6, 0.3)
    return post(frame, fi, shake=sh, grain=0.03)


def shot_card(b, fi):
    """3-01 章节卡（0—8 拍）。"""
    return chapter_card(b, fi, 0, "先驱")


SMOKE_COL = np.array([0.60, 0.54, 0.48], np.float32)
FIRE = np.array([1.0, 0.48, 0.12], np.float32)
NOZ_ALL = np.array(WANHU["nozzles"], np.float32)
NOZ = NOZ_ALL[NOZ_ALL[:, 1] > 840]          # 只用椅子下方最底一排喷口做推进火焰
NOZ_CX, NOZ_BOTTOM = float(NOZ[:, 0].mean()), float(NOZ[:, 1].max())


def flame_layer(dy, power, fi, subset=1.0):
    """椅下的推进火焰：底排每个喷口一道向下的焰柱（外焰橙、内核白黄），下方再合成一道推进光柱。"""
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
    small = cv2.GaussianBlur(small, (0, 0), 2.5)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) * 1.6
    gy = NOZ_BOTTOM + dy + 50
    glow = np.exp(-(((xx - NOZ_CX) ** 2) / (2 * 260 ** 2) + ((yy - gy) ** 2) / (2 * 150 ** 2)))[..., None]
    plume = np.exp(-(((xx - NOZ_CX) ** 2) / (2 * 110 ** 2)) - np.clip(yy - gy, 0, None) / (260 * max(power, 0.05)))
    plume = (plume * (yy > gy - 40))[..., None]
    return big + glow * FIRE * 0.55 * min(power, 1.2) + plume * (FIRE * 0.7 + 0.25) * min(power, 1.0)


def shift_layer(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def ground_smoke(level, rise, lit):
    """地面升起的尾烟：level 为烟层顶部离画面底边的高度，rise 让烟的纹理向上翻滚，lit 为火光照亮程度。"""
    n = np.roll(SMOKE[0], -int(rise) % H, axis=0) * 0.7 + np.roll(SMOKE[1], -int(rise * 0.6) % H, axis=0) * 0.3
    top = H - level
    m = np.clip((yy - top) / 160 + (n - 0.5) * 1.8, 0, 1)[..., None]
    col = SMOKE_COL + FIRE * 0.45 * lit * np.exp(-((xx - NOZ_CX) ** 2) / (2 * 420 ** 2))[..., None]
    return m, col


def shot_wanhu(r, fi):
    """3-02 万户（实际第 8—20 拍）：引信 → 喷口冒火星 → 第 16 拍点火，47 支火箭向下喷焰 →
    第 17 拍起人和椅子一起离地，加速飞出画面 → 尾烟升起盖满画面，转场到热气球。"""
    rb = min(r, 15.5) if r < 16 else r
    sky = WANHU["sky"]
    frame = sky.copy()

    # 会飞走的部分：点火后震动，第 17 拍起加速上升
    dx = dy = 0.0
    power = 0.0
    if r >= 16:
        power = min(1.0, 0.55 + (t_real(r) - t_real(16)) * 1.2) * (0.9 + 0.1 * np.sin(fi * 3.1))
        jr = np.random.default_rng(fi)
        dx, dy = jr.normal(0, 2.5), jr.normal(0, 2.5)
        if r >= 17:
            dy += -1500 * ease_in((r - 17) / 2.4, 2.3)
    fly_rgb, fly_a = WANHU["fly_rgb"], WANHU["fly_a"]
    if dx or dy:
        fly_rgb, fly_a = shift_layer(fly_rgb, dx, dy), shift_layer(fly_a[..., 0], dx, dy)[..., None]
    if power > 0:  # 火光照亮剪影下半部
        rim = np.exp(-(((xx - NOZ_CX) ** 2) / (2 * 300 ** 2) + ((yy - (NOZ_BOTTOM + dy - 60)) ** 2) / (2 * 140 ** 2)))[..., None]
        fly_rgb = fly_rgb + rim * FIRE * 0.55 * power
    frame = frame * (1 - fly_a) + fly_rgb * fly_a

    # 点火前：引信火花 → 分头烧向火箭 → 喷口零星冒火星
    if r < 16:
        p = np.clip((rb - 8.3) / 4.5, 0, 1)
        if p < 1:
            x, y = _along(WANHU["fuse"], p)
            spark(frame, x, y, fi)
        elif rb < 13.6:
            q = np.clip((rb - 12.8) / 0.8, 0, 1)
            hub = WANHU["fuse"][-1]
            for bx, by in WANHU["branches"]:
                spark(frame, lerp(hub[0], bx, q), lerp(hub[1], by, q), fi, 12, 1.1)
        else:
            k = np.clip((rb - 13.6) / 1.9, 0, 1)
            frame = frame + flame_layer(0, 0.12 + 0.25 * k, fi, subset=0.25 + 0.5 * k)
    else:
        frame = frame + flame_layer(dy, power, fi)

    # 地面与发射台
    frame = frame * (1 - WANHU["gnd_a"]) + WANHU["gnd_rgb"] * WANHU["gnd_a"]

    # 镜头：点火前慢推；起飞时略微向上跟
    zoom = lerp(1.0, 1.12, ease_io((rb - 8) / 8)) if r < 16 else 1.12
    cam_y = 600 + (0.12 * dy if r >= 17 else 0)
    m = cv2.getRotationMatrix2D((1262.0, float(cam_y)), 0, zoom)
    m[0, 2] += W / 2 + 300 - 1262
    m[1, 2] += H / 2 + 60 - cam_y
    frame = cv2.warpAffine(frame, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    caption(frame, r, 8.5, [("万户", "big"), ("47 支火箭，一把椅子", "mid"),
                            ("明朝 · 传说", "small"), ("今天，月球背面有一座以他命名的环形山", "small")], scrim=0.5)

    # 尾烟：点火后从地面升起，第 19.4—20 拍盖满画面，转场
    if r >= 16:
        el = t_real(r) - t_real(16)
        if r < 19.4:
            level = lerp(60, 380, ease_io((r - 16) / 3.4))
        else:
            level = lerp(380, H + 400, ease_in((r - 19.4) / 0.6, 1.6))
        sm, col = ground_smoke(level, el * 70, power)
        frame = frame * (1 - sm) + col * sm
    elif r >= 15.5:
        frame = frame * 0.8

    shake = (0.0, 0.0)
    if 16 <= r < 19.4:
        sr = np.random.default_rng(fi + 99)
        amp = 9 * (1 - 0.6 * np.clip((r - 17) / 2.4, 0, 1))
        shake = (sr.normal(0, amp), sr.normal(0, amp))
    return post(frame, fi, shake=shake, grain=0.035)


def shot_balloon(b, fi):
    """3-03 热气球（逻辑 16—24 拍）：万户的尾烟散开 → 从气球局部拉出整幅版画。"""
    src = DUO["balloon"]
    h, w = src.shape[:2]
    k = ease_io((b - 16) / 8)
    s0, s1 = 4.2 * K["balloon"], 0.80 * K["balloon"]
    scale = s0 * (s1 / s0) ** k
    cx, cy = lerp(0.53 * w, 0.5 * w, k), lerp(0.35 * h, 0.47 * h, k)
    sx = lerp(W / 2, 1210, k)
    rgb, a = place(src, cx, cy, scale, lerp(0, -1.2, k), sx, H / 2)
    frame = comp(PAPER_DARK.copy() * 1.6, rgb, a, shadow=0.6 * k)
    # 爆炸烟雾：第 16 拍起白烟铺满，1.5 拍内散开
    dt = t(b) - t(16)
    if dt < 1.3:
        thr = lerp(-0.2, 0.85, ease_io(dt / 1.3))
        sm = np.clip((SMOKE[0] * 0.7 + SMOKE[1] * 0.3 - thr) * 4, 0, 1)[..., None]
        frame = frame * (1 - sm) + SMOKE_COL * sm
    caption(frame, b, 17, [("1783.11.21 · PARIS", "date"), ("离开地面", "big"),
                           ("人类第一次乘热气球飞上天空", "mid"), ("孟格菲热气球 · 飞行约 25 分钟", "small")])
    return post(frame, fi)


def shot_lilienthal(b, fi):
    """3-06 李林塔尔（24—28 拍）：甩镜入画，向右上漂移。"""
    src = DUO["lilienthal"]
    h, w = src.shape[:2]
    sc = cover_scale(src) * 1.12
    k = (b - 24) / 4
    fr = (t(b) - t(24)) * FPS
    whip = 1 - ease_out_expo(fr / 7)
    cx, cy = lerp(0.55 * w, 0.62 * w, k), lerp(0.48 * h, 0.40 * h, k)
    rgb, a = place(src, cx, cy, sc * lerp(1.0, 1.08, k), lerp(1.5, -0.5, k), W / 2 + whip * 900, H / 2)
    frame = comp(PAPER_DARK.copy(), rgb, a)
    if whip > 0.02:
        frame = motion_blur(frame, 120 * whip, 0)
    caption(frame, b, 24.5, [("1891—1896", "date"), ("李林塔尔", "big"), ("近两千次滑翔", "mid"),
                             ("最后一次，他没能平安落地", "small")])
    return post(frame, fi)


def formula_img():
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


FORMULA = formula_img()


def draw_img(frame, rgb, a, x, y, alpha=1.0, reveal=1.0):
    a = a.copy() * alpha
    a[:, int(a.shape[1] * reveal):] = 0
    h, w = a.shape[:2]
    x0, y0 = int(x - w / 2), int(y - h / 2)
    reg = frame[y0:y0 + h, x0:x0 + w]
    frame[y0:y0 + h, x0:x0 + w] = reg * (1 - a) + rgb * a
    return frame


def shot_1903(b, fi):
    """3-07 1903 分屏（28—36 拍）：左莱特兄弟细节拆解，右齐奥尔科夫斯基手写方程。"""
    frame = PAPER_DARK.copy()
    src = DUO["wright"]
    h, w = src.shape[:2]
    # 左半屏：每 2 拍切一个局部，最后 4 拍是全图
    pc = cover_scale(src, W // 2, H)
    crops = [(28, 0.57, 0.37, 2.6 * K["wright"]), (30, 0.86, 0.44, 3.0 * K["wright"]), (32, 0.52, 0.45, pc * 1.08)]
    seg = [c for c in crops if b >= c[0]][-1]
    nxt = {28: 30, 30: 32, 32: 36}[seg[0]]
    k = (b - seg[0]) / (nxt - seg[0])
    rgb, a = place(src, seg[1] * w + lerp(-8, 8, k) * 3, seg[2] * h, seg[3] * lerp(1.0, 1.06, k), 0, W / 4, H / 2, W // 2, H)
    left = comp(PAPER_DARK[:, : W // 2].copy(), rgb, a)
    fl = flash_at(b, seg[0], 1) * 0.5 if seg[0] > 28 else 0
    left = left * (1 - fl) + fl
    left *= 1 - np.clip((yy[:, : W // 2, None] - H * 0.72) / (H * 0.28), 0, 1) * 0.75
    frame[:, : W // 2] = left
    # 右半屏：方程逐笔写出
    right = PAPER[:, W // 2:].copy() * 0.95
    frame[:, W // 2:] = right
    rv = np.clip((b - 28.5) / 4.5, 0, 1)
    draw_text(frame, "齐奥尔科夫斯基", "serif", 34, INK, W * 0.75, H * 0.36,
              min(max((b - 28.3) * 2, 0), 1), anchor="c", shadow=False, tracking=4)
    draw_img(frame, *FORMULA, W * 0.75, H * 0.5, 1.0, rv)
    if b > 34:
        draw_text(frame, "“地球是人类的摇篮，但人不能永远生活在摇篮里”", "serif", 25, INK, W * 0.75, H * 0.64,
                  min((b - 34) * 2, 1), anchor="c", shadow=False)
        draw_text(frame, "—— 1911 年书信", "serif", 22, INK, W * 0.75, H * 0.69, min((b - 34.3) * 2, 1), anchor="c", shadow=False)
    frame[:, : W // 2] *= 1 - np.clip(1 - yy[:, : W // 2, None] / (H * 0.3), 0, 1) * 0.6
    # 中线从上到下画出
    fr = (t(b) - t(28)) * FPS
    ln = int(H * ease_out_expo(fr / 8))
    frame[:ln, W // 2 - 2:W // 2 + 2] = np.array([0.85, 0.75, 0.55])
    # 顶部标题
    draw_text(frame, "1903，同一年", "brush", 110, CREAM, W * 0.25, 130, min(fr / 4, 1),
              lerp(1.25, 1, ease_out_expo(fr / 9)), anchor="c")
    if b > 28.5:
        a2 = min((b - 28.5) * 2, 1)
        draw_text(frame, "莱特兄弟 · 人类第一次驾驶飞机，飞了 12 秒", "serif", 34, CREAM, W * 0.25, H - 110, a2, anchor="c")
        draw_text(frame, "写下了飞出地球的公式", "serif", 34, INK, W * 0.75, H - 110, a2, anchor="c", shadow=False)
    sh, _ = shake_at(b, 28, 8, 0.3)
    return post(frame, fi, shake=sh)


def shot_nyt(b, fi):
    """3-08 纽约时报（36—40 拍）：报纸旋转冲入，第 38 拍急推到标题。"""
    src = NEWS
    fr = (t(b) - t(36)) * FPS
    k1 = ease_out_expo(fr / 9)
    scale = lerp(1.1, 0.52, k1)
    rot = lerp(-14, -4, k1)
    cx, cy = src.shape[1] / 2, src.shape[0] / 2
    sx = W * 0.62
    if b >= 38:
        k2 = ease_out_expo((t(b) - t(38)) * FPS / 7)
        scale = lerp(0.52, 1.05, k2) * lerp(1, 1.04, (b - 38) / 2)
        rot = lerp(-4, -1.5, k2)
        cx, cy = lerp(cx, NEWS_HEAD[0], k2), lerp(cy, NEWS_HEAD[1], k2)
        sx = lerp(W * 0.62, W * 0.74, k2)
    rgb, a = place(src, cx, cy, scale, rot, sx, H / 2 + 20)
    frame = comp(PAPER_DARK.copy(), rgb, a, shadow=0.7)
    caption(frame, b, 36.3, [("1920.01.13", "date"), ("嘲笑", "big"), ("《纽约时报》社论：", "mid"), ("火箭不可能在真空中飞行", "mid"),
                             ("被嘲笑的人，叫戈达德", "small")], scrim=0.8)
    sh, e = shake_at(b, 38 if b >= 38 else 36, 14, 0.35)
    return post(frame, fi, flash=flash_at(b, 38, 1) * 0.6, shake=sh, chroma=4 * e)


def shot_goddard(b, fi):
    """3-09 戈达德（40—44 拍）：从人物向上摇到火箭。"""
    src = DUO["goddard"]
    h, w = src.shape[:2]
    sc = cover_scale(src)
    k = ease_io((b - 40) / 4)
    cy = lerp(0.66 * h, 0.27 * h, k)
    rgb, a = place(src, w / 2, cy, sc * lerp(1.0, 1.05, k))
    frame = comp(PAPER_DARK.copy(), rgb, a)
    caption(frame, b, 40.3, [("1926.03.16", "date"), ("戈达德", "big"), ("第一枚液体燃料火箭", "mid"),
                             ("飞了 2.5 秒 · 高约 12.5 米", "small")], scrim=0.75)
    sh, _ = shake_at(b, 40, 10, 0.3)
    return post(frame, fi, flash=flash_at(b, 40, 2) * 0.8, shake=sh)


def shot_v2(b, fi):
    """3-11 V-2 拍下的地球（44—48 拍）：从黑场和噪点中浮现，缓慢旋转。"""
    src = DUO["v2"]
    sc = cover_scale(src)
    k = (b - 44) / 4
    rgb, a = place(src, src.shape[1] / 2, src.shape[0] / 2, sc * lerp(1.04, 1.12, k), lerp(-3, 0, k))
    emerge = ease_io(min((b - 44) / 2.0, 1))
    frame = rgb * emerge
    noise = cv2.resize(GRAIN_SMALL[(fi * 7) % 12], (W, H))[..., None]
    frame = np.clip(frame + noise * 0.18 * (1 - emerge * 0.7), 0, 1)
    caption(frame, b, 44.8, [("1946.10.24", "date"), ("从太空看见地球", "big"), ("人类第一次", "mid"),
                             ("拍下它的，是一枚战争武器 V-2 · 约 105 公里高空", "small")])
    return post(frame, fi, grain=0.06)


def shot_x1(b, fi):
    """3-12 X-1（48—52 拍）：加速推进 + 速度线 → 第 50 拍音爆 → 余震漂移。"""
    src = DUO["x1"]
    h, w = src.shape[:2]
    sc = cover_scale(src)
    if b < 50:
        k = ease_in((b - 48) / 2, 2.6)
        scale = sc * lerp(1.05, 1.75, k)
        cx = lerp(0.50 * w, 0.22 * w, k)
        rgb, a = place(src, cx, lerp(0.48, 0.44, k) * h, scale)
        frame = comp(PAPER_DARK.copy(), rgb, a)
        frame = motion_blur(frame, 6 + 90 * k, 0)
        # 速度线
        n = int(10 + 50 * k)
        lines = np.zeros((H, W), np.float32)
        r = np.random.default_rng(fi)
        for _ in range(n):
            y = int(r.uniform(0, H))
            x0 = int(r.uniform(-200, W))
            cv2.line(lines, (x0, y), (x0 + int(r.uniform(200, 700)), y), 1.0, 1)
        frame = frame + cv2.GaussianBlur(lines, (0, 0), 0.8)[..., None] * 0.5 * k
        return post(frame, fi)
    k = (b - 50) / 2
    rgb, a = place(src, 0.52 * w, 0.5 * h, sc * lerp(1.0, 1.04, k), lerp(0.8, 0, k))
    frame = comp(PAPER_DARK.copy(), rgb, a)
    caption(frame, b, 50.15, [("1.06 马赫", "big"), ("突破音障", "mid"), ("1947.10.14 · 耶格尔 · 贝尔 X-1", "small")],
            x=110, y=700, scrim=0.7)
    sh, e = shake_at(b, 50, 34, 0.9)
    return post(frame, fi, flash=flash_at(b, 50, 3), shake=sh, chroma=9 * e)


def shot_black(b, fi):
    """第 52 拍：一拍完全黑场静音。"""
    return post(np.zeros((H, W, 3), np.float32), fi, grain=0.02, vignette=False, weave=False)


def shot_next(b, fi):
    """第 53 拍起：下一章"竞赛"的章节卡随重音落下（不冲出）。"""
    return chapter_card(b, fi, 53 - 0.3, "竞赛", exit_zoom=False)


SHOTS = [(0, 8, shot_card), (16, 24, shot_balloon), (24, 28, shot_lilienthal),
         (28, 36, shot_1903), (36, 40, shot_nyt), (40, 44, shot_goddard), (44, 48, shot_v2),
         (48, 52, shot_x1), (52, 53, shot_black), (53, 999, shot_next)]


def render_frame(fi):
    r = beat_at(fi / FPS)          # 实际拍号
    if 8 <= r < WANHU_END:
        return shot_wanhu(r, fi)
    b = r if r < 8 else r - SHIFT  # 其余镜头按逻辑拍号编写
    for b0, b1, fn in SHOTS:
        if b0 <= b < b1:
            return fn(b, fi)
    return np.zeros((H, W, 3), np.float32)


def main():
    OUT.mkdir(exist_ok=True)
    n = int(DURATION * FPS)
    only = [int(a) for a in sys.argv[1:]]
    if only:  # 只渲染指定帧做检查
        for fi in only:
            img = (render_frame(fi) * 255).astype(np.uint8)
            Image.fromarray(img).save(OUT / f"frame_{fi:04d}.jpg", quality=88)
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(OUT / "music.wav"), "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest",
           str(OUT / "ch3_demo.mp4")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(n):
        p.stdin.write((render_frame(fi) * 255).astype(np.uint8).tobytes())
        if fi % 100 == 0:
            print(f"{fi}/{n}", flush=True)
    p.stdin.close()
    p.wait()
    print("done", OUT / "ch3_demo.mp4")


if __name__ == "__main__":
    main()
