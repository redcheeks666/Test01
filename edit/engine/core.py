"""《天问》渲染引擎核心：画面工具、文字与字幕、章节卡、年份标尺、后期。

坐标系 1920×1080，颜色为 float32 RGB [0, 1]。各章节文件只调用这里的函数。
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
FONTS = ROOT / "edit" / "fonts"
W, H, FPS = 1920, 1080, 30

INK = (0.16, 0.11, 0.07)
CREAM = (0.99, 0.96, 0.89)
GOLD = (0.86, 0.72, 0.42)
SEAL_RED = (0.72, 0.13, 0.10)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_rng = np.random.default_rng(3)


# ---------------- 数学 ----------------

def clamp01(x):
    return float(np.clip(x, 0, 1))


def lerp(a, b, k):
    return a + (b - a) * k


def ease_io(x):
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_expo(x):
    x = clamp01(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_in(x, p=2.5):
    return clamp01(x) ** p


def ease_out(x, p=2.5):
    return 1 - (1 - clamp01(x)) ** p


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


# ---------------- 纹理 ----------------

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


def make_paper(dark=False, seed=None):
    n = fbm(H, W, 6, seed=(11 if dark else 5) if seed is None else seed)
    fib = cv2.GaussianBlur((_rng.random((H, W)) > 0.996).astype(np.float32), (0, 0), 0.8)
    fib = cv2.filter2D(fib, -1, np.ones((1, 9), np.float32) / 9)
    if dark:
        base = np.array([0.11, 0.085, 0.065], np.float32)
        p = base * (0.8 + 0.4 * n[..., None]) + fib[..., None] * 0.05
    else:
        base = np.array([0.91, 0.86, 0.75], np.float32)
        p = base * (0.88 + 0.16 * n[..., None]) - fib[..., None] * 0.15
    return np.clip(p, 0, 1).astype(np.float32)


PAPER = make_paper()
PAPER_DARK = make_paper(dark=True)
SMOKE = [fbm(H, W, 6, s) for s in (21, 22)]
VIGNETTE = np.clip(1 - 0.55 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2), 0, 1)[..., None]
GRAIN = [_rng.standard_normal((H // 2, W // 2)).astype(np.float32) for _ in range(12)]
GRAD_L = (np.clip(1 - xx / (W * 0.55), 0, 1) ** 1.6)[..., None]
GRAD_B = (np.clip((yy - H * 0.62) / (H * 0.38), 0, 1) ** 1.4)[..., None]


def starfield(seed=1, density=900, dark=(0.02, 0.025, 0.05)):
    r = np.random.default_rng(seed)
    img = np.ones((H, W, 3), np.float32) * np.array(dark, np.float32)
    img *= (0.8 + 0.4 * fbm(H, W, 5, seed + 3))[..., None]
    for _ in range(density):
        x, y = int(r.uniform(0, W)), int(r.uniform(0, H))
        b = r.uniform(0.15, 0.9) ** 2
        rad = 1 if r.random() > 0.04 else 2
        cv2.circle(img, (x, y), rad, (b, b * 0.97, b * 0.9), -1)
    return img


# ---------------- 几何与合成 ----------------

def place(src, cx, cy, scale, rot=0.0, sx=W / 2, sy=H / 2, out_w=W, out_h=H, border=cv2.BORDER_CONSTANT):
    """把 src 上的点 (cx, cy) 放到屏幕 (sx, sy)，缩放 scale、旋转 rot 度。返回 (rgb, alpha)。"""
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), rot, scale)
    m[0, 2] += sx - cx
    m[1, 2] += sy - cy
    rgb = cv2.warpAffine(src, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=border)
    a = cv2.warpAffine(np.ones(src.shape[:2], np.float32), m, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return rgb, a[..., None]


def cam(src, cx, cy, scale, rot=0.0, sx=W / 2, sy=H / 2):
    """虚拟镜头：边缘复制填充，不留黑边。"""
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), rot, scale)
    m[0, 2] += sx - cx
    m[1, 2] += sy - cy
    return cv2.warpAffine(src, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def comp(bg, rgb, a, shadow=0.0):
    if shadow:
        sh = cv2.GaussianBlur(a[..., 0], (0, 0), 18)
        sh = np.roll(np.roll(sh, 14, 0), 10, 1)[..., None]
        bg = bg * (1 - sh * shadow)
    return bg * (1 - a) + rgb * a


def cover_scale(src, w=W, h=H):
    return max(w / src.shape[1], h / src.shape[0])


def fit_scale(src, w=W, h=H):
    return min(w / src.shape[1], h / src.shape[0])


def motion_blur(img, length, angle=0.0):
    length = int(length)
    if length < 2:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1.0), (length, length))
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def sharpen(img, amt=0.6, sigma=1.6):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(img + (img - blur) * amt, 0, 1)


def shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def glow(x, y, size, color, strength=1.0, sx=None, sy=None):
    sx = sx or size
    sy = sy or size
    g = np.exp(-(((xx - x) ** 2) / (2 * sx ** 2) + ((yy - y) ** 2) / (2 * sy ** 2)))[..., None]
    return g * np.array(color, np.float32) * strength


def circle_mask(cx, cy, r, soft=2.0):
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip((r - d) / soft + 0.5, 0, 1)[..., None]


# ---------------- 调色 ----------------

def gray(img):
    return img @ np.array([0.3, 0.59, 0.11], dtype=np.float32)


def duotone(img, dark=(0.09, 0.06, 0.04), light=(0.94, 0.88, 0.76), gamma=1.0, contrast=1.2):
    g = np.clip((gray(img) - 0.5) * contrast + 0.5, 0, 1) ** gamma
    d, l_ = np.array(dark, np.float32), np.array(light, np.float32)
    return (d + g[..., None] * (l_ - d)).astype(np.float32)


def grade_bw(img, contrast=1.25):
    """1957—1972：黑白胶片。"""
    return duotone(img, (0.05, 0.05, 0.05), (0.93, 0.92, 0.88), 1.0, contrast)


def grade_faded(img, amt=0.55):
    """1977—2003：褪色胶片彩色。"""
    g = gray(img)[..., None]
    out = img * amt + g * (1 - amt)
    out = out * np.array([1.06, 1.0, 0.86], np.float32) + np.array([0.04, 0.03, 0.02], np.float32)
    out = (out - 0.5) * 0.92 + 0.5
    return np.clip(out, 0, 1).astype(np.float32)


def grade_clean(img, cool=0.0, contrast=1.05, sat=1.05):
    g = gray(img)[..., None]
    out = g + (img - g) * sat
    out = (out - 0.5) * contrast + 0.5
    if cool:
        out = out * np.array([1 - 0.06 * cool, 1.0, 1 + 0.08 * cool], np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


def grade_teal_orange(img, amt=0.35):
    """序章与伍：冷色夜景，保留火焰暖色。"""
    g = gray(img)[..., None]
    shadows = np.array([0.80, 0.95, 1.12], np.float32)
    highs = np.array([1.10, 1.0, 0.86], np.float32)
    tint = shadows * (1 - g) + highs * g
    out = img * (1 - amt) + img * tint * amt
    out = (out - 0.5) * 1.08 + 0.5
    return np.clip(out, 0, 1).astype(np.float32)


# ---------------- 文字 ----------------

FONT_FILES = {"brush": "MaShanZheng-Regular.ttf", "serif": "NotoSerifSC[wght].ttf",
              "mono": "IBMPlexMono-Regular.ttf", "wild": "ZhiMangXing-Regular.ttf"}


@lru_cache(None)
def font(kind, size, weight=600):
    f = ImageFont.truetype(str(FONTS / FONT_FILES[kind]), size)
    if kind == "serif":
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


@lru_cache(None)
def text_img(text, kind, size, color, tracking=0, weight=600):
    f = font(kind, size, weight)
    pad = size // 2
    widths = [f.getbbox(ch)[2] - f.getbbox(ch)[0] if ch != " " else size // 3 for ch in text]
    tw = sum(widths) + tracking * max(len(text) - 1, 0) + 2 * pad
    th = int(size * 1.45) + 2 * pad
    im = Image.new("L", (max(tw, 1), th), 0)
    d = ImageDraw.Draw(im)
    x = pad
    for ch, w in zip(text, widths):
        d.text((x - f.getbbox(ch)[0], pad), ch, font=f, fill=255)
        x += w + tracking
    a = np.asarray(im, np.float32)[..., None] / 255.0
    rgb = np.ones((th, max(tw, 1), 3), np.float32) * np.array(color, np.float32)
    return rgb, a, pad


def text_width(text, kind, size, tracking=0):
    rgb, a, pad = text_img(text, kind, size, (1, 1, 1), tracking)
    return a.shape[1] - 2 * pad


def draw_text(frame, text, kind, size, color, x, y, alpha=1.0, scale=1.0, tracking=0,
              shadow=True, reveal=1.0, anchor="lt", weight=600):
    """x, y 为文字左上角（anchor='lt'）或中心（anchor='c'）。reveal 控制从左到右显现。"""
    if alpha <= 0.001 or not text:
        return frame
    rgb, a, pad = text_img(text, kind, size, tuple(color), tracking, weight)
    if reveal < 1:
        a = a.copy()
        a[:, int(pad + (a.shape[1] - 2 * pad) * reveal):] = 0
    if scale != 1.0:
        nw, nh = max(1, int(a.shape[1] * scale)), max(1, int(a.shape[0] * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)[..., None]
    h, w = a.shape[:2]
    if anchor == "c":
        x0, y0 = int(x - w / 2), int(y - h / 2)
    elif anchor == "rt":
        x0, y0 = int(x - w + pad * scale), int(y - pad * scale)
    else:
        x0, y0 = int(x - pad * scale), int(y - pad * scale)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xb <= xa or yb <= ya:
        return frame
    sa = a[ya - y0:yb - y0, xa - x0:xb - x0] * alpha
    sr = rgb[ya - y0:yb - y0, xa - x0:xb - x0]
    region = frame[ya:yb, xa:xb]
    if shadow:
        light = sum(color) > 1.5
        sh = cv2.GaussianBlur(sa[..., 0], (0, 0), 7)[..., None]
        tight = cv2.GaussianBlur(sa[..., 0], (0, 0), 1.8)[..., None]
        if light:
            region = region * (1 - np.clip(sh * 0.9 + tight * 1.2, 0, 0.92))
        else:
            region = region * (1 - sh * 0.15) + sh * 0.12
    frame[ya:yb, xa:xb] = region * (1 - sa) + sr * sa
    return frame


def draw_img(frame, rgb, a, x, y, alpha=1.0, reveal=1.0, anchor="c"):
    a = a * alpha
    if reveal < 1:
        a = a.copy()
        a[:, int(a.shape[1] * reveal):] = 0
    h, w = a.shape[:2]
    x0, y0 = (int(x - w / 2), int(y - h / 2)) if anchor == "c" else (int(x), int(y))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xb <= xa or yb <= ya:
        return frame
    sa = a[ya - y0:yb - y0, xa - x0:xb - x0]
    sr = rgb[ya - y0:yb - y0, xa - x0:xb - x0] if rgb.ndim == 3 else rgb
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - sa) + sr * sa
    return frame


CAP_SPEC = {"big": ("brush", 132, 0, 158), "mid": ("serif", 46, 2, 70), "small": ("serif", 31, 2, 50),
            "date": ("mono", 32, 3, 54), "lead": ("serif", 30, 6, 52)}


def caption(frame, c, at, lines, color=CREAM, x=110, y=150, scrim=None, out=None, anchor="lt"):
    """三级文字。at 为镜头内的拍号；lines = [(文字, 级别)]，级别 big / mid / small / date / lead。
    大字在 at 冲入，其余依次延后约半拍。浅色字自动在左侧压暗渐变底。out 给定时在该拍淡出。"""
    if c.lb < at:
        return frame
    if scrim is None:
        scrim = 0.72 if sum(color) > 1.5 and anchor == "lt" else 0.0
    fade = 1.0
    if out is not None and c.lb > out:
        fade = 1 - clamp01((c.lb - out) * 3)
        if fade <= 0:
            return frame
    if scrim:
        # 背景越亮，暗底越重：保证浅色字下方的平均亮度不超过约 0.3
        h_blk = sum(CAP_SPEC[l][3] for _, l in lines) + 40
        reg = frame[max(int(y) - 20, 0):min(int(y + h_blk), H), max(int(x) - 40, 0):int(x) + 820]
        lum = float(gray(reg).mean()) if reg.size else 0.0
        need = (1 - 0.30 / lum) / 0.8 if lum > 0.30 else 0.0
        scrim = float(np.clip(max(scrim, need), 0, 0.92))
        k = clamp01(c.since(at) * FPS / 6) * fade
        frame *= 1 - GRAD_L * scrim * k
    cy = y
    delay = 0.0
    for text, lvl in lines:
        kind, size, tr, step = CAP_SPEC[lvl]
        bi = at + delay
        if c.lb >= bi:
            fr = c.since(bi) * FPS
            xx_ = x if anchor == "lt" else x
            anc = "lt" if anchor == "lt" else "c"
            if lvl == "big":
                a = min(fr / 4, 1) * fade
                s = lerp(1.28, 1.0, ease_out_expo(fr / 9))
                if text.isascii():          # 纯英文大标题用衬线体，比毛笔体的拉丁字母更庄重
                    draw_text(frame, text, "serif", 112, color, xx_, cy + 6, a, s, 6, anchor=anc, weight=700)
                else:
                    draw_text(frame, text, kind, size, color, xx_, cy, a, s, tr, anchor=anc)
            elif lvl == "date":
                draw_text(frame, text, kind, size, color, xx_, cy, fade, 1.0, tr,
                          reveal=min(1, int(fr) / max(len(text), 1)), anchor=anc)
            else:
                a = min(fr / 8, 1) * fade
                draw_text(frame, text, kind, size, color, xx_, cy + (1 - ease_out_expo(fr / 12)) * 22, a, 1.0, tr,
                          anchor=anc)
        cy += step
        delay += 0.5 if lvl in ("big", "date") else 0.35
    return frame


# ---------------- 章节卡 ----------------

def _seal(size=96, text="天问"):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([4, 4, size - 4, size - 4], radius=10, fill=(184, 34, 26, 255))
    f = font("brush", int(size * 0.4))
    for k, ch in enumerate(text):
        d.text((size / 2, size * (0.3 + 0.4 * k) + 2), ch, font=f, fill=(245, 228, 210, 0), anchor="mm")
    arr = np.asarray(im, np.float32) / 255.0
    arr[..., 3] *= (fbm(size, size, 4, 31) > 0.38).astype(np.float32)
    return arr[..., :3], arr[..., 3:]


SEAL = _seal()


def chapter_card(c, name, exit_zoom=True, size=300):
    """统一章节卡：纸本底纹，只显示章节名（毛笔大字），镜头第 2.5 拍盖"天问"印，最后半拍冲进墨迹。"""
    frame = PAPER.copy()
    lb = c.lb
    s_ = lerp(1.06, 1.0, ease_io((lb - 0.5) / max(c.beats - 1, 1)))
    ink_a = clamp01(c.since(0.3) * FPS / 10)
    sz = int(size * s_) if len(name) <= 2 else int(size * 0.62 * s_)
    draw_text(frame, name, "brush", sz, INK, W / 2, H / 2 - 20, ink_a, 1.0, tracking=30 if len(name) <= 2 else 10,
              anchor="c", shadow=False)
    tw = text_width(name, "brush", sz, 30 if len(name) <= 2 else 10)
    if lb >= 2.5:
        fr3 = c.since(2.5) * FPS
        ss = lerp(1.5, 1.0, ease_out_expo(fr3 / 5))
        rgb, a = SEAL
        s2 = int(96 * ss)
        rgb_s, a_s = cv2.resize(rgb, (s2, s2)), cv2.resize(a, (s2, s2))[..., None]
        x0, y0 = int(W / 2 + tw / 2 + 40 - s2 / 2), int(H / 2 + sz * 0.35 - s2 / 2)
        k = min(fr3 / 2, 1)
        reg = frame[y0:y0 + s2, x0:x0 + s2]
        frame[y0:y0 + s2, x0:x0 + s2] = reg * (1 - a_s * k) + rgb_s * a_s * k
    if exit_zoom and lb > c.beats - 0.5:
        z = ease_in((lb - (c.beats - 0.5)) / 0.5, 2)
        frame, _ = place(frame, W / 2, H / 2 - 20, lerp(1.0, 2.4, z))
        frame = motion_blur(frame, 40 * z, 90)
    sh, _ = shake_at(c, 2.5, 6, 0.3)
    return post(frame, c.fi, shake=sh, grain=0.03)


# ---------------- 年份标尺（先驱、竞赛） ----------------

class YearTicker:
    """左下角的年份：每个事件到来时年份数字快速滚动到新年份，下面一条时间轴上的光点同步前进。
    keys = [(章节内拍号, 年份或文字, 小字)]，年份为 int 时滚动，为 str 时直接显示。"""

    def __init__(self, keys, y0=1780, y1=2026, roll_beats=1.2, x=110, y=H - 168, color=CREAM):
        self.keys = sorted(keys, key=lambda k: k[0])
        self.y0, self.y1, self.roll = y0, y1, roll_beats
        self.x, self.y, self.color = x, y, color

    def _state(self, b):
        cur = None
        prev = None
        for k in self.keys:
            if b >= k[0]:
                prev, cur = cur, k
        return prev, cur

    def draw(self, frame, b, alpha=1.0):
        prev, cur = self._state(b)
        if cur is None or alpha <= 0:
            return frame
        year, sub = cur[1], cur[2]
        k = clamp01((b - cur[0]) / self.roll)
        shown, pos = year, year
        if isinstance(year, int):
            if prev is not None and isinstance(prev[1], int) and k < 1:
                e = ease_out(k, 3)
                v = prev[1] + (year - prev[1]) * e
                shown = int(round(v))
                pos = v
            shown = str(shown)
        else:
            pos = self.y0 - 30
        x, y = self.x, self.y
        # 暗底，保证在任何画面上可读
        reg = frame[y - 30:H - 40, x - 60:x + 520]
        gy, gx = np.mgrid[0:reg.shape[0], 0:reg.shape[1]].astype(np.float32)
        m = np.clip(1 - gx / reg.shape[1], 0, 1) ** 1.2 * 0.55 * alpha
        frame[y - 30:H - 40, x - 60:x + 520] = reg * (1 - m[..., None])
        fk, fs, ft = ("mono", 66, 4) if shown.lstrip("-").isdigit() else ("serif", 54, 6)
        draw_text(frame, shown, fk, fs, self.color, x, y + (0 if fk == "mono" else 6), alpha, tracking=ft)
        if sub:
            draw_text(frame, sub, "serif", 26, self.color, x + text_width(shown, fk, fs, ft) + 22, y + 34,
                      alpha * 0.9, tracking=2)
        # 时间轴
        ly = y + 104
        x1 = x + 440
        frame[ly:ly + 2, x:x1] = frame[ly:ly + 2, x:x1] * (1 - 0.6 * alpha) + np.array(self.color) * 0.6 * alpha
        for yr in range(1800, self.y1 + 1, 50):
            tx = int(x + (yr - self.y0) / (self.y1 - self.y0) * (x1 - x))
            frame[ly - 6:ly, tx:tx + 2] = frame[ly - 6:ly, tx:tx + 2] * (1 - 0.6 * alpha) + np.array(self.color) * 0.6 * alpha
        p = clamp01((pos - self.y0) / (self.y1 - self.y0))
        px = int(x + p * (x1 - x))
        frame[ly - 1:ly + 3, x:px] = np.array(GOLD) * alpha + frame[ly - 1:ly + 3, x:px] * (1 - alpha)
        cv2.circle(frame, (px, ly + 1), 6, tuple(float(v) for v in GOLD), -1)
        return frame


# ---------------- 后期 ----------------

def post(frame, fi, flash=0.0, shake=(0, 0), chroma=0.0, grain=0.045, vignette=True, weave=True, vig_amt=0.65):
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
        frame = frame * ((1 - vig_amt) + vig_amt * VIGNETTE)
    if grain:
        g = cv2.resize(GRAIN[fi % len(GRAIN)], (W, H), interpolation=cv2.INTER_LINEAR)
        frame = frame + g[..., None] * grain
    if flash > 0:
        frame = frame * (1 - flash) + flash
    return np.clip(frame, 0, 1).astype(np.float32)


def shake_at(c, at, amp, decay_s=0.8):
    dt = c.since(at)
    if dt < 0:
        return (0.0, 0.0), 0.0
    e = np.exp(-dt / decay_s * 4)
    return (amp * e * np.sin(dt * 67), amp * e * np.cos(dt * 53)), e


def flash_at(c, at, frames=3, peak=1.0):
    fr = c.since(at) * FPS
    if 0 <= fr < frames:
        return peak * (1.0 - fr / frames * 0.4)
    if frames <= fr < frames + 5:
        return peak * 0.6 * (1 - (fr - frames) / 5)
    return 0.0


def fade_black(frame, k):
    return frame * (1 - clamp01(k))


def smoke_wipe(frame, k, color=(0.60, 0.54, 0.48), reverse=False):
    """烟雾转场：k 从 0 到 1，烟从无到满（reverse=True 时从满到散）。"""
    thr = lerp(0.85, -0.2, ease_io(k)) if not reverse else lerp(-0.2, 0.85, ease_io(k))
    sm = np.clip((SMOKE[0] * 0.7 + SMOKE[1] * 0.3 - thr) * 4, 0, 1)[..., None]
    return frame * (1 - sm) + np.array(color, np.float32) * sm
