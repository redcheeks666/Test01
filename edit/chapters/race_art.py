"""竞赛章的代码绘制：斯普特尼克、莱卡、出舱的航天员、水墨长征火箭、月船 3 号着陆器、黑白地球。

苏联档案照片授权不明确，这几样用剪影与线描画出来；画风统一为黑白档案海报感。
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from engine.core import W, H, fbm, grade_bw, circle_mask, xx, yy
from engine.media import image

S = 2


def _rgba(im, size):
    im = im.resize(size, Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    return a[..., :3], a[..., 3:]


@lru_cache(None)
def earth_bw(max_edge=2400):
    """蓝色弹珠转黑白，作为苏联段落的地球。"""
    return grade_bw(image("NA-12", "02", max_edge), 1.15)


def sputnik(size=520, ang=-25):
    """抛光金属球 + 四根后掠天线。"""
    n = size * S
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = n * 0.42
    r = n * 0.13
    a0 = np.radians(ang)
    for k, spread in enumerate((-0.30, -0.12, 0.10, 0.28)):
        a = a0 + np.pi + spread
        L = n * (0.46 if k in (0, 3) else 0.54)
        x1, y1 = c + np.cos(a) * L, c + np.sin(a) * L
        d.line([(c, c), (x1, y1)], fill=(210, 210, 205, 255), width=max(2, int(n * 0.006)))
    yy_, xx_ = np.mgrid[0:n, 0:n].astype(np.float32)
    dd = np.sqrt((xx_ - c) ** 2 + (yy_ - c) ** 2)
    sphere = np.clip(r - dd + 1, 0, 1)
    light = np.clip(1 - np.sqrt((xx_ - (c - r * 0.35)) ** 2 + (yy_ - (c - r * 0.35)) ** 2) / (r * 1.6), 0, 1)
    shade = (0.25 + 0.75 * light ** 1.2) * 255
    arr = np.asarray(im, np.float32)
    for ch in range(3):
        arr[..., ch] = arr[..., ch] * (1 - sphere) + shade * sphere
    arr[..., 3] = np.maximum(arr[..., 3], sphere * 255)
    im = Image.fromarray(arr.astype(np.uint8), "RGBA")
    return _rgba(im, (size, size))


def laika(size=520):
    """坐姿小狗剪影（尖耳、长吻），放在舱窗圆框里。"""
    n = size * S
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = n / 400

    def P(pts):
        return [(x * k, y * k) for x, y in pts]
    body = [(150, 42), (176, 92), (196, 88), (206, 44), (214, 100), (236, 130), (262, 172), (300, 214), (318, 262),
            (322, 318), (340, 372), (300, 376), (268, 344), (226, 318), (182, 318), (172, 372), (148, 376),
            (134, 300), (128, 250), (136, 196), (120, 166), (82, 156), (58, 140), (62, 128), (100, 112), (132, 98)]
    d.polygon(P(body), fill=(235, 232, 225, 255))
    tail = [(318, 300), (352, 260), (372, 214), (360, 210), (340, 250), (312, 284)]
    d.polygon(P(tail), fill=(235, 232, 225, 255))
    d.ellipse(P([(118, 112), (128, 122)]), fill=(30, 30, 30, 255))   # 眼
    return _rgba(im, (size, size))


def cosmonaut(size=520):
    """出舱的航天员：白色舱外服、圆形头盔、背包，向左伸手。"""
    n = size * S
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = n / 400
    W_, G_, D_ = (238, 236, 230, 255), (170, 168, 160, 255), (40, 40, 44, 255)

    def P(pts):
        return [(x * k, y * k) for x, y in pts]
    d.rounded_rectangle(P([(215, 120), (275, 250)]), radius=14 * k, fill=G_)                 # 背包
    d.rounded_rectangle(P([(160, 130), (250, 270)]), radius=30 * k, fill=W_, outline=G_, width=int(3 * k))
    d.ellipse(P([(165, 60), (245, 140)]), fill=W_, outline=G_, width=int(3 * k))              # 头盔
    d.ellipse(P([(178, 78), (226, 122)]), fill=D_)                                              # 面窗
    d.ellipse(P([(186, 84), (202, 98)]), fill=(120, 130, 150, 255))
    d.line(P([(170, 160), (110, 120), (70, 104)]), fill=W_, width=int(26 * k), joint="curve")   # 伸出的手臂
    d.line(P([(240, 170), (275, 230), (262, 280)]), fill=W_, width=int(24 * k), joint="curve")
    d.line(P([(185, 262), (170, 330), (196, 380)]), fill=W_, width=int(30 * k), joint="curve")
    d.line(P([(228, 262), (250, 320), (240, 372)]), fill=W_, width=int(30 * k), joint="curve")
    for x, y in ((70, 104), (262, 280), (196, 380), (240, 372)):
        d.ellipse(P([(x - 14, y - 14), (x + 14, y + 14)]), fill=G_)
    return _rgba(im, (size, size))


def long_march(height=900):
    """水墨风格的长征二号 F：芯级、四个助推器、整流罩、逃逸塔。"""
    w = int(height * 0.36)
    im = Image.new("RGBA", (w * S, height * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ink = (22, 18, 14, 255)
    cx = w / 2
    k = height / 900

    def P(pts):
        return [(cx * S + x * k * S, y * k * S) for x, y in pts]
    d.line(P([(0, 0), (0, 70)]), fill=ink, width=int(6 * k * S))                         # 逃逸塔
    d.polygon(P([(-34, 70), (34, 70), (40, 230), (-40, 230)]), fill=ink)                  # 整流罩
    d.rectangle([P([(-40, 230)])[0], P([(40, 820)])[0]], fill=ink)                        # 芯级
    for sx in (-1, 1):
        for off in (0, 1):
            x0 = sx * (40 + off * 30)
            d.polygon(P([(x0, 560), (x0 + sx * 30, 560), (x0 + sx * 30, 840), (x0, 840)]), fill=ink)
            d.polygon(P([(x0, 560), (x0 + sx * 15, 510), (x0 + sx * 30, 560)]), fill=ink)
    d.rectangle([P([(-36, 400)])[0], P([(36, 412)])[0]], fill=(200, 40, 30, 255))          # 一道红色标记
    im = im.resize((w, height), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    # 墨色飞白：边缘做一点干笔纹理
    tex = (fbm(height, w, 5, 77) > 0.42).astype(np.float32)
    a[..., 3] *= 0.96 + 0.04 * tex
    return a[..., :3], a[..., 3:]


def lander(size=360):
    """着陆器剪影：方形机身、四条斜腿、太阳翼。"""
    n = size * S
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = n / 400
    G_ = (220, 210, 170, 255)

    def P(pts):
        return [(x * k, y * k) for x, y in pts]
    d.rectangle(P([(130, 170), (270, 260)]), fill=G_)
    d.rectangle(P([(160, 140), (240, 170)]), fill=(180, 170, 140, 255))
    d.polygon(P([(270, 180), (380, 150), (390, 190), (270, 215)]), fill=(70, 80, 110, 255))
    for x0, x1 in ((140, 90), (180, 160), (220, 240), (260, 310)):
        d.line(P([(x0, 250), (x1, 330)]), fill=G_, width=int(8 * k))
        d.ellipse(P([(x1 - 14, 326), (x1 + 14, 338)]), fill=G_)
    return _rgba(im, (size, size))


def paste(frame, rgb, a, cx, cy, scale=1.0, alpha=1.0, rot=0.0):
    import cv2
    if scale != 1.0 or rot:
        h, w = a.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
        nw, nh = int(w * scale * 1.5) + 2, int(h * scale * 1.5) + 2
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        rgb = cv2.warpAffine(rgb, m, (nw, nh))
        a = cv2.warpAffine(a[..., 0], m, (nw, nh))[..., None]
    h, w = a.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xb <= xa or yb <= ya:
        return frame
    sa = a[ya - y0:yb - y0, xa - x0:xb - x0] * alpha
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - sa) + rgb[ya - y0:yb - y0, xa - x0:xb - x0] * sa
    return frame
