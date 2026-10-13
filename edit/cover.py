"""《天问》封面：16:9、9:16、3:4、4:3 四个比例。

沿用账号封面的版式：深蓝星空底、粗黑体白色大标题、红底白字斜标签、白色四角星、带标题栏的圆角窗口。
画面取片中星舰点火的实拍帧（SpaceX），配片中的月亮和"天问"印章。

python3 cover.py            # 输出到 ../delivery/covers/
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine.core import SEAL  # noqa: E402
from chapters.prologue import _moon  # noqa: E402

OUT = HERE.parent / "delivery" / "covers"
SANS = str(HERE / "fonts" / "NotoSansSC[wght].ttf")
HERO = HERE / ".cache" / "clips" / "sx0102_pad_v2" / "00200.jpg"   # 星舰点火（SX-01/02，SpaceX）
RED = (227, 38, 43)
NAVY0, NAVY1 = np.array([10, 18, 78]), np.array([34, 52, 150])


def font(size, weight=900):
    f = ImageFont.truetype(SANS, int(size))
    f.set_variation_by_axes([weight])
    return f


def background(w, h, seed=7):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w * 0.5) / w) ** 2 + ((yy - h * 0.35) / h) ** 2)
    k = np.clip(1 - d * 1.5, 0, 1)[..., None]
    bg = NAVY0 * (1 - k) + NAVY1 * k
    r = np.random.default_rng(seed)
    n = int(w * h / 900)
    xs, ys = r.integers(0, w, n), r.integers(0, h, n)
    v = r.random(n) ** 3 * 255
    bg[ys, xs] = np.maximum(bg[ys, xs], v[:, None])
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def sparkle(d, cx, cy, r, fill=(255, 255, 255)):
    pts = []
    for i in range(8):
        a = np.pi / 2 * (i // 2) + (np.pi / 4 if i % 2 else 0)
        rr = r if i % 2 == 0 else r * 0.16
        pts.append((cx + rr * np.cos(a), cy - rr * np.sin(a)))
    d.polygon(pts, fill=fill)


def moon(img, cx, cy, r):
    m, a, mr = _moon()
    s = r / mr
    h, w = m.shape[:2]
    mm = Image.fromarray((np.clip(m * np.array([1.0, 0.97, 0.9]), 0, 1) * 255).astype(np.uint8)).resize(
        (int(w * s), int(h * s)), Image.LANCZOS)
    ma = Image.fromarray((a * 255).astype(np.uint8)).resize(mm.size, Image.LANCZOS)
    glow = Image.new("L", img.size, 0)
    ImageDraw.Draw(glow).ellipse([cx - r * 1.35, cy - r * 1.35, cx + r * 1.35, cy + r * 1.35], fill=90)
    glow = glow.filter(ImageFilter.GaussianBlur(r * 0.35))
    img.paste(Image.new("RGB", img.size, (200, 200, 255)), (0, 0), glow)
    img.paste(mm, (int(cx - mm.width / 2), int(cy - mm.height / 2)), ma)


def window(img, box, hero_crop, radius, border, bar):
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = x1 - x0, y1 - y0
    shadow = Image.new("L", img.size, 0)
    ImageDraw.Draw(shadow).rounded_rectangle([x0 + 10, y0 + 18, x1 + 10, y1 + 18], radius, fill=150)
    img.paste((0, 0, 0), (0, 0), shadow.filter(ImageFilter.GaussianBlur(18)))
    frame = Image.new("RGB", (w, h), (24, 24, 28))
    src = Image.open(HERO).convert("RGB")
    cx, cy, zoom = hero_crop
    ih = h - bar
    s = max(w / src.width, ih / src.height) * zoom
    sw, sh = int(src.width * s), int(src.height * s)
    src = src.resize((sw, sh), Image.LANCZOS)
    left = int(np.clip(cx * sw - w / 2, 0, sw - w))
    top = int(np.clip(cy * sh - ih / 2, 0, sh - ih))
    frame.paste(src.crop((left, top, left + w, top + ih)), (0, bar))
    d = ImageDraw.Draw(frame)
    for i, c in enumerate(((255, 95, 87), (254, 188, 46), (40, 200, 64))):
        r = bar * 0.17
        d.ellipse([bar * 0.55 + i * bar * 0.55 - r, bar / 2 - r, bar * 0.55 + i * bar * 0.55 + r, bar / 2 + r], fill=c)
    f = font(bar * 0.42, 500)
    d.text((w / 2, bar / 2), "天问 · 星舰入轨", font=f, fill=(200, 200, 205), anchor="mm")
    live = "2026.09.28"
    fl = font(bar * 0.42, 700)
    tw = d.textlength(live, font=fl)
    d.ellipse([w - bar * 0.5 - tw - bar * 0.5, bar / 2 - bar * 0.12, w - bar * 0.5 - tw - bar * 0.26, bar / 2 + bar * 0.12],
              fill=(40, 200, 64))
    d.text((w - bar * 0.5, bar / 2), live, font=fl, fill=(235, 235, 240), anchor="rm")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=255)
    ImageDraw.Draw(img).rounded_rectangle([x0 - border, y0 - border, x1 + border, y1 + border], radius + border,
                                          fill=(245, 245, 250))
    img.paste(frame, (x0, y0), mask)


def label(img, text, x, y, size, rot=-3.0, anchor="l"):
    """红底白字标签（带深色投影，略倾斜）。x, y 为左上角（anchor='l'）。"""
    f = font(size)
    tw = ImageDraw.Draw(img).textlength(text, font=f)
    pad_x, pad_y = size * 0.35, size * 0.18
    w, h = int(tw + pad_x * 2), int(size * 1.3 + pad_y * 2)
    tile = Image.new("RGBA", (w + 24, h + 24), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rectangle([12 + 8, 12 + 8, 12 + w + 8, 12 + h + 8], fill=(90, 0, 0, 200))
    d.rectangle([12, 12, 12 + w, 12 + h], fill=RED + (255,))
    d.text((12 + w / 2, 12 + h / 2), text, font=f, fill=(255, 255, 255, 255), anchor="mm")
    tile = tile.rotate(-rot, resample=Image.BICUBIC, expand=True)
    if anchor == "r":
        x -= tile.width
    img.paste(tile, (int(x), int(y)), tile)
    return tile.size


def text(img, s, x, y, size, anchor="la", stroke=0):
    d = ImageDraw.Draw(img)
    f = font(size)
    d.text((x + size * 0.04, y + size * 0.06), s, font=f, fill=(5, 8, 40), anchor=anchor)
    d.text((x, y), s, font=f, fill=(255, 255, 255), anchor=anchor, stroke_width=stroke, stroke_fill=(255, 255, 255))
    return d.textbbox((x, y), s, font=f, anchor=anchor)


def seal(img, x, y, size):
    rgb, a = SEAL
    s = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    m = Image.fromarray((np.clip(a[..., 0] if a.ndim == 3 else a, 0, 1) * 255).astype(np.uint8)).resize((size, size))
    img.paste(s, (int(x), int(y)), m)


def portrait(w, h):
    img = background(w, h)
    d = ImageDraw.Draw(img)
    u = w / 1080
    moon(img, w * 0.86, h * 0.085 + 40 * u, 130 * u)
    lab_h = label(img, "「Vibe知识大赏」", 54 * u, h * 0.035, 54 * u, rot=-2.5)[1]
    ty = h * 0.035 + lab_h + 4 * u
    bb = text(img, "天问", 60 * u, ty, 230 * u)
    seal(img, bb[2] + 18 * u, bb[3] - 120 * u, int(92 * u))
    bb2 = text(img, "人类向往星空的五千年！", 62 * u, bb[3] + 26 * u, 78 * u)
    top = bb2[3] + 50 * u
    bottom = h - 120 * u
    window(img, (52 * u, top, w - 52 * u, bottom), (0.5, 0.5, 1.0), int(30 * u), int(7 * u), int(52 * u))
    label(img, "从嫦娥奔月到星舰入轨", 34 * u, bottom - 70 * u, 50 * u, rot=-3)
    for sx, sy, r in ((0.93, 0.25, 26), (0.07, 0.30, 14), (0.95, 0.60, 16), (0.70, 0.04, 12)):
        sparkle(d, w * sx, h * sy if sy > 0.2 else sy * h + 20 * u, r * u)
    return img


def landscape(w, h):
    img = background(w, h)
    d = ImageDraw.Draw(img)
    u = h / 1080
    lab_w, lab_h = label(img, "「Vibe知识大赏」", 50 * u, 46 * u, 48 * u, rot=-2.5)
    title = "天问，人类向往星空的五千年！"
    room = w - 56 * u - 300 * u                     # 右侧留给月亮
    size = min(112 * u, room / ImageDraw.Draw(img).textlength(title, font=font(100)) * 100)
    moon(img, w - 160 * u, 150 * u, 105 * u)
    bb = text(img, title, 56 * u, 46 * u + lab_h + 6 * u, size)
    top = bb[3] + 46 * u
    bottom = h - 60 * u
    margin = 60 * u
    window(img, (margin, top, w - margin, bottom), (0.5, 0.66 if w / h < 1.5 else 0.45, 1.0), int(28 * u), int(6 * u), int(46 * u))
    label(img, "从嫦娥奔月到星舰入轨", 34 * u, bottom - 74 * u, 50 * u, rot=-3)
    for sx, sy, r in ((0.965, 0.32, 22), (0.03, 0.30, 12), (0.80, 0.05, 14)):
        sparkle(d, w * sx, h * sy, r * u)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (w, h) in {"16x9": (1920, 1080), "4x3": (1440, 1080), "9x16": (1080, 1920), "3x4": (1080, 1440)}.items():
        img = landscape(w, h) if w > h else portrait(w, h)
        p = OUT / f"tianwen_cover_{name}.png"
        img.save(p, optimize=True)
        print(p, img.size)


if __name__ == "__main__":
    main()
