"""《纽约时报》版面重绘（示意，非原版扫描）：1920 年嘲笑戈达德的社论，1969 年的更正。"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine.core import FONTS, fbm


def _font(name, size, weight=None):
    f = ImageFont.truetype(str(FONTS / name), size)
    if weight:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


def newspaper(date_line, head_lines, seed=4):
    """返回 (图像, 标题框中心)。版式两次完全一致，只换日期和标题，方便前后呼应。"""
    pw, ph = 1500, 1900
    im = Image.new("RGB", (pw, ph), (226, 214, 186))
    d = ImageDraw.Draw(im)
    ink = (34, 26, 20)
    serif = _font("NotoSerifSC[wght].ttf", 118, 900)
    small = _font("IBMPlexMono-Regular.ttf", 28)
    head = _font("NotoSerifSC[wght].ttf", 70, 800)
    d.text((pw / 2, 120), "The New York Times", font=serif, fill=ink, anchor="mm")
    d.line([(60, 210), (pw - 60, 210)], fill=ink, width=4)
    d.text((pw / 2, 240), date_line, font=small, fill=ink, anchor="mm")
    d.line([(60, 268), (pw - 60, 268)], fill=ink, width=2)
    r = np.random.default_rng(seed)
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
    d.rectangle([bx0, 310, bx1, 800], outline=ink, width=3)
    for i, line in enumerate(head_lines):
        d.text(((bx0 + bx1) / 2, 380 + i * 90), line, font=head, fill=ink, anchor="mm")
    d.line([(bx0 + 60, 530), (bx1 - 60, 530)], fill=ink, width=2)
    for i in range(10):
        lw = (bx1 - bx0 - 80) * (1 if i < 9 else 0.5)
        d.rectangle([bx0 + 40, 560 + i * 22, bx0 + 40 + lw, 569 + i * 22], fill=(90, 80, 68))
    arr = np.asarray(im, np.float32) / 255.0
    arr = arr * (0.9 + 0.12 * fbm(ph, pw, 5, 9)[..., None])
    return np.clip(arr, 0, 1).astype(np.float32), ((bx0 + bx1) / 2, 450)
