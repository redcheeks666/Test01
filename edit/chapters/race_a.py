"""肆 · 竞赛（前半，120 BPM）：1957—1990。左下角年份标尺接着先驱章继续走。

4A 苏联"第一"连发（黑白档案风，代码绘制）→ 4B 地出（颜色在这一刻回来）、阿波罗 11 号、纽约时报更正、登月脚印 →
4C 1970 年 4 月分屏、联盟 11 号、阿波罗 17 号 → 4D 旅行者金唱片、航天飞机、挑战者号、暗淡蓝点（全程静音）、哈勃。
"""
import cv2
import numpy as np

from engine.core import (W, H, FPS, CREAM, INK, GOLD, PAPER, PAPER_DARK, chapter_card, post, place, lerp, ease_io,
                         ease_in, ease_out, ease_out_expo, clamp01, glow, caption, grade_bw, grade_faded, grade_clean,
                         draw_text, starfield, circle_mask, flash_at, shake_at, fbm, xx, yy, YearTicker, cover_scale,
                         motion_blur, SMOKE)
from engine.film import Chapter, Shot
from engine.media import Clip, image, retime
from engine.shots import ken, video, split, memorial
from engine import music as M
from chapters import race_art as A
from chapters.newsprint import newspaper

A11 = Clip("na09_a11", "NA/NA-09/NA-09_01_apollo11_70mm_launch_nasa.mp4", 15.0, 36.0, crop=(0.06, 0.06, 0.94, 0.94),
           size=(1600, 1600))
A11_T = retime([(0, 1.0), (1.0, 1.0), (1.6, 1.8), (6.0, 3.2)], start=0.3)
SPACE = starfield(21, 800, (0.015, 0.015, 0.02))
_c = {}


def img(aid, cand="01", grade=None, max_edge=3000):
    key = (aid, cand, grade.__name__ if grade else None)
    if key not in _c:
        x = image(aid, cand, max_edge)
        _c[key] = grade(x) if grade else x
    return _c[key]


def bw_earth_limb(frame, cy=H + 1150, r=1350, alpha=1.0):
    e = A.earth_bw()
    s = 2 * r / (0.94 * e.shape[1])
    rgb, a = place(e, e.shape[1] / 2, e.shape[0] / 2, s, 0, W / 2, cy)
    m = circle_mask(W / 2, cy, r, 3.0) * a * alpha
    frame = frame * (1 - m) + rgb * m
    d = np.sqrt((xx - W / 2) ** 2 + (yy - cy) ** 2)
    haze = np.exp(-((d - r) / 30) ** 2)[..., None] * 0.35 * alpha
    return frame + haze


def s_card(c):
    return chapter_card(c, "竞赛")


# ---------------- 4A ----------------

_ENV = None


def _sputnik_env():
    global _ENV
    if _ENV is None:
        x = M._decode("AU/AU-01/AU-01_01_sputnik_signal_nasa.mp3")
        hop = M.SR // FPS
        n = len(x) // hop
        _ENV = np.abs(x[: n * hop]).reshape(n, hop).max(axis=1)
    return _ENV


SPUT = A.sputnik(520)


def s_sputnik(c):
    f = bw_earth_limb(SPACE.copy())
    x = lerp(900, 1500, c.k)
    y = lerp(470, 400, c.k)
    A.paste(f, *SPUT, x, y, 1.0, 1.0, rot=lerp(0, -8, c.k))
    f = f + glow(x - 20, y - 20, 18, (1, 1, 1), 0.4 * (0.6 + 0.4 * np.sin(c.lsec * 9)))
    # 示波器：随真实信号跳动
    env = _sputnik_env()
    i = int((c.since(0.5)) * FPS)
    amp = env[i] if 0 <= i < len(env) else 0.0
    x0, x1, yc = 760, 1300, 930
    f[yc - 50:yc + 50, x0:x1] *= 0.35
    ts = np.linspace(0, 1, x1 - x0)
    wave = np.sin(ts * 2 * np.pi * 18 + c.lsec * 30) * amp * 40 * np.sin(ts * np.pi)
    for k, (xp, yp) in enumerate(zip(range(x0, x1), (yc + wave).astype(int))):
        cv2.circle(f, (xp, int(yp)), 1, (0.95, 0.93, 0.85), -1)
    f[yc - 50, x0:x1] = 0.5
    f[yc + 50, x0:x1] = 0.5
    caption(f, c, 0.5, [("斯普特尼克 1 号", "big"), ("第一颗人造卫星", "mid"),
                        ("你听到的“滴—滴”，是它从太空发回的真实信号", "small")])
    return post(grade_bw(f, 1.05), c.fi, grain=0.06)


LAIKA = A.laika(460)


def s_laika(c):
    f = SPACE.copy()
    cx, cy, r = 1250, 560, 300
    inner = np.ones((H, W, 3), np.float32) * 0.12 + glow(cx - 80, cy - 120, 260, (0.3, 0.3, 0.3), 0.6)
    m = circle_mask(cx, cy, r, 2)
    f = f * (1 - m) + inner * m
    A.paste(f, *LAIKA, cx + 10, cy + 40, lerp(0.95, 1.02, c.k))
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rim = np.clip(1 - np.abs(d - r - 14) / 16, 0, 1)[..., None]
    f = f * (1 - rim) + 0.55 * rim
    for k in range(12):
        a = k * np.pi / 6
        cv2.circle(f, (int(cx + (r + 14) * np.cos(a)), int(cy + (r + 14) * np.sin(a))), 6, (0.3, 0.3, 0.3), -1)
    caption(f, c, 0.3, [("莱卡", "big"), ("第一个进入轨道的生命", "mid"), ("它没有回来", "small")])
    return post(grade_bw(f), c.fi, grain=0.06)


def s_luna2(c):
    f = SPACE.copy()
    moon = img("NA-01", "01", grade_bw, 2400)
    s = 760 / (0.94 * moon.shape[0])
    rgb, a = place(moon, moon.shape[1] / 2, moon.shape[0] / 2, s * lerp(1.0, 1.06, c.k), 0, 1350, 560)
    m = circle_mask(1350, 560, 380 * lerp(1.0, 1.06, c.k), 2) * a
    f = f * (1 - m) + rgb * m
    p = ease_in(clamp01(c.lb / 2.5), 1.6)
    px, py = lerp(300, 1180, p), lerp(120, 470, p)
    if c.lb < 2.5:
        for k in range(14):
            q = max(p - k * 0.012, 0)
            f = f + glow(lerp(300, 1180, q), lerp(120, 470, q), 5, (1, 1, 1), 0.5 * (1 - k / 14))
    f = f + glow(1180, 470, 60, (1, 1, 1), 1.2 * flash_at(c, 2.5, 3, 1.0))
    caption(f, c, 0.3, [("月球 2 号", "big"), ("人造物体第一次抵达月球", "mid")])
    sh, _ = shake_at(c, 2.5, 6, 0.3)
    return post(grade_bw(f), c.fi, grain=0.06, shake=sh)


def _orbit(f, cx, cy, rx, ry, tilt, k, col=(0.92, 0.9, 0.84), width=2, dot=True):
    th = np.linspace(0, 2 * np.pi * k, max(2, int(240 * k)))
    ct, st = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
    pts = np.stack([cx + rx * np.cos(th) * ct - ry * np.sin(th) * st, cy + rx * np.cos(th) * st + ry * np.sin(th) * ct], 1)
    cv2.polylines(f, [pts.astype(np.int32)], False, col, width, cv2.LINE_AA)
    if dot and k > 0:
        cv2.circle(f, tuple(pts[-1].astype(int)), 7, col, -1)
    return f


def _earth_disk(f, cx, cy, r):
    e = A.earth_bw()
    s = 2 * r / (0.94 * e.shape[1])
    rgb, a = place(e, e.shape[1] / 2, e.shape[0] / 2, s, 0, cx, cy)
    m = circle_mask(cx, cy, r, 2) * a
    return f * (1 - m) + rgb * m


def s_gagarin(c):
    f = _earth_disk(SPACE.copy(), 1250, 560, 280)
    k = ease_io(clamp01((c.lb - 0.5) / 6.0))
    f = _orbit(f, 1250, 560, 440, 150, -14, k)
    mins = int(108 * k)
    draw_text(f, f"{mins:3d}", "mono", 48, CREAM, 1520, 860, 1.0, anchor="c")
    draw_text(f, "分钟", "serif", 34, CREAM, 1610, 864, 1.0, anchor="c")
    caption(f, c, 0.4, [("加加林", "big"), ("第一个进入太空的人", "mid"), ("绕地球一圈", "small")])
    return post(grade_bw(f), c.fi, grain=0.06)


def s_tereshkova(c):
    f = _earth_disk(SPACE.copy(), 1250, 560, 210)
    n = int(48 * ease_out(clamp01(c.lb / 3.2), 1.6))
    for i in range(n):
        f = _orbit(f, 1250, 560, 330 + (i % 6) * 4, 110 + (i % 5) * 6, -60 + i * 2.6, 1.0, (0.7, 0.7, 0.66), 1, False)
    draw_text(f, f"{n:2d}", "mono", 48, CREAM, 1530, 860, 1.0, anchor="c")
    draw_text(f, "圈", "serif", 34, CREAM, 1600, 864, 1.0, anchor="c")
    caption(f, c, 0.3, [("捷列什科娃", "big"), ("第一位进入太空的女性", "mid"), ("绕地球 48 圈", "small")])
    return post(grade_bw(f), c.fi, grain=0.06)


COSMO = A.cosmonaut(520)


def s_leonov(c):
    f = bw_earth_limb(SPACE.copy(), cy=H + 900, r=1250)
    cv2.rectangle(f, (0, 300), (150, 640), (0.25, 0.25, 0.27), -1)
    cv2.circle(f, (150, 470), 90, (0.18, 0.18, 0.2), -1)
    bx, by = lerp(1150, 1200, c.k), lerp(470, 440, c.k)
    t_ = np.linspace(0, 1, 60)
    tx = lerp(160, bx + 60, t_)
    ty = 470 + np.sin(t_ * np.pi) * 120 + (by - 470) * t_
    cv2.polylines(f, [np.stack([tx, ty], 1).astype(np.int32)], False, (0.8, 0.8, 0.76), 3, cv2.LINE_AA)
    A.paste(f, *COSMO, bx, by, 1.0, 1.0, rot=lerp(-6, 8, c.k))
    caption(f, c, 0.3, [("列昂诺夫", "big"), ("第一次太空行走", "mid"), ("在舱外约 12 分钟", "small")])
    return post(grade_bw(f), c.fi, grain=0.06)


# ---------------- 4B ----------------

def s_earthrise(c):
    src = image("NA-11", "01", 3000)
    f = ken(c, src, (0.5, 0.60, 2.1), (0.5, 0.46, 1.7))
    col = ease_io(clamp01((c.lb - 3.0) / 2.0))
    f = grade_bw(f) * (1 - col) + grade_clean(f, 0, 1.05, 1.1) * col
    caption(f, c, 0.6, [("地出", "big"), ("人类第一次从月球轨道看见地球升起", "mid"), ("阿波罗 8 号", "small")])
    return post(f, c.fi, grain=0.05)


def s_apollo11(c):
    st = A11_T(c.lsec)
    z = lerp(1.0, 1.12, ease_io(c.k))
    f = video(c, A11, st, grade=lambda x: grade_faded(x, 0.75), zoom=z, cx=0.47, cy=lerp(0.62, 0.45, ease_io(c.k)))
    caption(f, c, 2.4, [("阿波罗 11 号", "big"), ("目标：月球", "mid"), ("阿波罗，希腊神话中的太阳神", "small")])
    sh, e = shake_at(c, 2.0, 22, 1.0)
    return post(f, c.fi, flash=flash_at(c, 2.0, 2, 0.8), shake=sh, chroma=6 * e, grain=0.05)


NEWS69, NEWS69_HEAD = newspaper("NEW YORK, THURSDAY, JULY 17, 1969", ["A CORRECTION", ""])


def s_nyt69(c):
    fr = c.since(0) * FPS
    k1 = ease_out_expo(fr / 9)
    scale, rot = lerp(1.1, 0.52, k1), lerp(-14, -4, k1)
    cx, cy, sx = NEWS69.shape[1] / 2, NEWS69.shape[0] / 2, W * 0.62
    if c.lb >= 2:
        k2 = ease_out_expo(c.since(2) * FPS / 7)
        scale = lerp(0.52, 1.05, k2) * lerp(1, 1.04, (c.lb - 2) / 4)
        rot = lerp(-4, -1.5, k2)
        cx, cy = lerp(cx, NEWS69_HEAD[0], k2), lerp(cy, NEWS69_HEAD[1], k2)
        sx = lerp(W * 0.62, W * 0.74, k2)
    rgb, a = place(NEWS69, cx, cy, scale, rot, sx, H / 2 + 20)
    sh_a = cv2.GaussianBlur(a[..., 0], (0, 0), 18)
    f = PAPER_DARK * (1 - np.roll(np.roll(sh_a, 14, 0), 10, 1)[..., None] * 0.7)
    f = f * (1 - a) + rgb * a
    caption(f, c, 0.3, [("更正", "big"), ("《纽约时报》：", "mid"), ("火箭可以在真空中飞行", "mid"),
                        ("本报对此错误表示遗憾 · 距那篇社论 49 年", "small")], scrim=0.8)
    sh, e = shake_at(c, 2 if c.lb >= 2 else 0, 14, 0.35)
    return post(f, c.fi, flash=flash_at(c, 2, 1, 0.6), shake=sh, chroma=4 * e)


def s_landing(c):
    if c.lb < 6:
        src = img("NA-10", "02", lambda x: grade_faded(x, 0.8))
        crops = [(0, (0.47, 0.29, 3.2)), (2, (0.50, 0.47, 2.2)), (4, (0.5, 0.5, 1.0))]
        t0, (cx, cy, z) = [cr for cr in crops if c.lb >= cr[0]][-1]
        f = ken(c, src, (cx, cy, z), (cx, cy - 0.01, z * 1.06), t0=t0, t1=t0 + 2)
        fl = flash_at(c, t0, 1, 0.5) if t0 > 0 else 0
        f = f * (1 - fl) + fl
    else:
        src = img("NA-10", "01", lambda x: grade_faded(x, 0.8))
        f = ken(c, src, (0.5, 0.5, 1.0), (0.64, 0.70, 1.45), t0=6, t1=22)
    caption(f, c, 8.0, [("这枚脚印", "big"), ("至今还在", "mid"), ("导航计算机内存约 4KB", "small")])
    return post(f, c.fi, grain=0.05)


# ---------------- 4C ----------------

def _ink_reveal(k, seed=0):
    n = SMOKE[seed % 2]
    thr = lerp(1.1, -0.1, ease_io(k))
    return np.clip((n - thr) * 5, 0, 1)[..., None]


def s_split70(c):
    left = ken(c, img("NA-21", "01", lambda x: grade_faded(x, 0.7)), (0.5, 0.5, 1.05, -2), (0.5, 0.5, 1.15, 3),
               out_w=W // 2)
    left = np.pad(left, ((0, 0), (0, W - W // 2), (0, 0)))
    right = PAPER.copy()
    dfh = image("CN-01", "01", 1400)
    dfh = dfh[: int(dfh.shape[0] * 0.92)]            # 裁掉右下角水印
    rgb, a = place(dfh, dfh.shape[1] / 2, dfh.shape[0] / 2, 620 / dfh.shape[1] * lerp(1.0, 1.06, c.k),
                   lerp(0, 4, c.k), W * 0.75, H * 0.48)
    rev = _ink_reveal(clamp01((c.lb - 0.5) / 2.0), 1)
    right = right * (1 - a * rev) + rgb * a * rev
    right_half = np.zeros_like(right)
    right_half[:, : W // 2] = right[:, W // 2:]
    f = split(left, right_half, line=ease_out_expo(c.since(0) * FPS / 8))
    f[: int(H * 0.25), : W // 2] *= np.linspace(0.35, 1, int(H * 0.25))[:, None, None]
    fr = c.since(0) * FPS
    draw_text(f, "1970 年 4 月，同一个月", "brush", 70, CREAM, W * 0.25, 120, min(fr / 4, 1),
              lerp(1.25, 1, ease_out_expo(fr / 9)), anchor="c")
    a2 = clamp01((c.lb - 1.0) * 2)
    f[int(H * 0.66):int(H * 0.76), : W // 2] *= 0.45
    draw_text(f, "阿波罗 13 号 · 险些回不了家", "serif", 34, CREAM, W * 0.25, H * 0.71, a2, anchor="c")
    draw_text(f, "东方红一号 · 让太空响起《东方红》", "serif", 34, INK, W * 0.75, H * 0.80, a2, anchor="c", shadow=False)
    return post(f, c.fi, grain=0.04)


def s_soyuz11(c):
    return memorial(c, None, "1971 · 礼炮 1 号与联盟 11 号",
                    ["格奥尔基·多布罗沃利斯基", "弗拉季斯拉夫·沃尔科夫", "维克托·帕察耶夫"],
                    sub="第一座空间站。返回途中，三名航天员遇难", hold=0.0)


def _crt(f, fi):
    f = f.copy()
    f[::3] *= 0.75
    f = cv2.GaussianBlur(f, (0, 0), 1.2)
    return f * (0.92 + 0.08 * np.sin(fi * 1.7))


def s_apollo17(c):
    if c.lb < 6:
        f = ken(c, img("NA-12", "01", lambda x: grade_faded(x, 0.7)), (0.5, 0.5, 1.0), (0.55, 0.42, 1.25), t0=0, t1=6)
        f = _crt(f, c.fi)
    else:
        z = lerp(1.6, 0.55, ease_out(clamp01((c.lb - 6) / 6), 2))
        src = img("NA-12", "02", lambda x: grade_faded(x, 0.8))
        rgb, a = place(src, src.shape[1] / 2, src.shape[0] / 2, cover_scale(src) * z)
        f = SPACE * 0.4 * (1 - a) + rgb * a
        f = f * (1 - clamp01((c.lb - 10.5) / 1.5))
    caption(f, c, 0.5, [("最后一次", "big"), ("人类最后一次离开月球", "mid"), ("之后半个多世纪，再没有人去过", "small")])
    return post(f, c.fi, grain=0.05)


# ---------------- 4D ----------------

def s_voyager(c):
    seq = [(0, "01", (0.5, 0.6, 1.0), (0.5, 0.4, 1.15)), (2, "02", (0.5, 0.5, 1.1), (0.45, 0.45, 1.25)),
           (4, "03", (0.5, 0.5, 1.05), (0.55, 0.5, 1.2)), (6, "04", (0.5, 0.42, 1.0), (0.5, 0.45, 1.12))]
    t0, cand, a_, b_ = [s for s in seq if c.lb >= s[0]][-1]
    t1 = t0 + 2 if t0 < 6 else 12
    f = ken(c, img("NA-13", cand, lambda x: grade_faded(x, 0.7)), a_, b_, t0=t0, t1=t1)
    fl = flash_at(c, t0, 1, 0.4) if t0 > 0 else 0
    f = f * (1 - fl) + fl
    caption(f, c, 0.4, [("旅行者号", "big"), ("一封可能永远没人读的信", "mid"), ("金唱片 · 55 种语言的问候", "small")])
    if c.lb >= 6.5:
        a = clamp01((c.lb - 6.5) * 1.5)
        f[int(H * 0.70):int(H * 0.82)] *= 1 - 0.55 * a
        draw_text(f, "“太空朋友，恁好，恁食饱未？有闲着来阮遮坐哦”", "serif", 36, CREAM, W / 2 + 120, H * 0.745, a, anchor="c")
        draw_text(f, "金唱片里的闽南语问候", "serif", 24, (0.85, 0.82, 0.75), W / 2 + 120, H * 0.80, a, anchor="c")
    return post(f, c.fi, grain=0.05)


def s_sts1(c):
    f = ken(c, img("NA-22", "01", lambda x: grade_faded(x, 0.75)), (0.5, 0.80, 1.05), (0.5, 0.38, 1.12))
    caption(f, c, 0.4, [("航天飞机", "big"), ("第一次升空", "mid"), ("这一天，距加加林进入太空整整 20 年", "small")])
    sh, _ = shake_at(c, 0, 8, 0.4)
    return post(f, c.fi, grain=0.05, shake=sh)


def s_challenger(c):
    return memorial(c, img("NA-23", "01"), "1986 · 挑战者号",
                    ["迪克·斯科比", "迈克尔·史密斯", "朱迪丝·雷斯尼克", "埃里森·鬼冢", "罗纳德·麦克奈尔",
                     "格雷戈里·贾维斯", "克丽斯塔·麦考利夫"], sub="七名航天员遇难", hold=4.0)


PBD = (0.5946, 0.5195)


def s_pbd(c):
    if c.lb < 1.0:
        f = np.zeros((H, W, 3), np.float32)
    else:
        src = image("NA-14", "02", 3000)
        k = ease_in(clamp01((c.lb - 1.0) / 11.0), 1.6)
        cx, cy = lerp(0.5, PBD[0], ease_io(clamp01((c.lb - 1.0) / 6))), lerp(0.5, PBD[1], ease_io(clamp01((c.lb - 1.0) / 6)))
        f = ken(c, src, (cx, cy, 1.0), (cx, cy, 1.0)) if False else None
        h, w = src.shape[:2]
        z = cover_scale(src) * (1.0 * (7.0 / 1.0) ** k)
        m = cv2.getRotationMatrix2D((cx * w, cy * h), 0, z)
        m[0, 2] += W / 2 - cx * w
        m[1, 2] += H / 2 - cy * h
        f = cv2.warpAffine(src, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        f = grade_clean(f, 0, 1.08, 1.0) * clamp01((c.lb - 1.0) / 1.0)
    if c.lb >= 4.5:
        draw_text(f, "60 亿公里外回望", "serif", 40, CREAM, W / 2, H * 0.80, clamp01((c.lb - 4.5) / 1.0), anchor="c",
                  tracking=8)
    if c.lb >= 9.0:
        draw_text(f, "这是我们的全部", "brush", 96, CREAM, W / 2, H * 0.22, clamp01((c.lb - 9.0) / 1.0), anchor="c",
                  tracking=16)
    return post(f, c.fi, grain=0.03, vignette=False)


def s_hubble(c):
    f = ken(c, img("NA-24", "01", lambda x: grade_faded(x, 0.85)), (0.5, 0.5, 1.05), (0.56, 0.46, 1.2))
    caption(f, c, 0.4, [("哈勃", "big"), ("把望远镜送上天", "mid"), ("1993 年，宇航员上天为它修好了“近视”", "small")])
    return post(f, c.fi, grain=0.04)


SHOTS = [Shot(8, s_card, name="4-01"), Shot(8, s_sputnik, name="4-02"), Shot(4, s_laika, name="4-03"),
         Shot(4, s_luna2, name="4-04"), Shot(8, s_gagarin, name="4-05"), Shot(4, s_tereshkova, name="4-06"),
         Shot(6, s_leonov, name="4-07"), Shot(8, s_earthrise, name="4-08"), Shot(12, s_apollo11, name="4-09"),
         Shot(6, s_nyt69, name="4-10"), Shot(22, s_landing, name="4-11"), Shot(10, s_split70, name="4-12"),
         Shot(8, s_soyuz11, name="4-13"), Shot(12, s_apollo17, name="4-14"), Shot(12, s_voyager, name="4-15"),
         Shot(6, s_sts1, name="4-16"), Shot(10, s_challenger, name="4-17"), Shot(16, s_pbd, name="4-18"),
         Shot(6, s_hubble, name="4-19")]
ST = np.concatenate([[0], np.cumsum([s.beats for s in SHOTS])])

TICKER = YearTicker([(ST[1], 1957, "10.04"), (ST[2], 1957, "11.03"), (ST[3], 1959, "09"), (ST[4], 1961, "04.12"),
                     (ST[5], 1963, "06.16"), (ST[6], 1965, "03.18"), (ST[7], 1968, "12.24"), (ST[8], 1969, "07.16"),
                     (ST[9], 1969, "07.17"), (ST[10], 1969, "07.20"), (ST[11], 1970, "04"), (ST[12], 1971, "06.30"),
                     (ST[13], 1972, "12.14"), (ST[14], 1977, "09.05"), (ST[15], 1981, "04.12"), (ST[16], 1986, "01.28"),
                     (ST[17], 1990, "02.14"), (ST[18], 1990, "04.24")], roll_beats=1.6)


def overlay(img_, c):
    if c.b < ST[1]:
        return img_
    a = clamp01((c.b - ST[1]) * 2)
    if ST[17] <= c.b < ST[18]:          # 暗淡蓝点：标尺淡出，让画面只剩那一个像素
        a *= 1 - clamp01((c.b - ST[17]) * 2) + clamp01((c.b - (ST[18] - 0.5)) * 2)
    return TICKER.draw(img_, c.b, a)


# ---------------- 配乐 ----------------

DM = [50, 53, 57]


def music(sc, ch):
    T, t = ch.T, ch.t

    def drums(b0, b1, gain=1.0, hats=True, snare_on=True):
        b = b0
        while b < b1 - 1e-6:
            if b % 1 == 0:
                sc.add("bgm", M.kick(1.0), T(b), 0.55 * gain)
            if snare_on and b % 2 == 1:
                sc.add("bgm", M.snare(0.9), T(b), 0.32 * gain)
            if hats:
                sc.add("bgm", M.hat(0.8 if b % 1 else 0.5), T(b), 0.25 * gain)
            b += 0.5

    def bass(b0, b1, roots, gain=0.32):
        b = b0
        while b < b1 - 1e-6:
            r = roots[int((b - b0) // 4) % len(roots)]
            pat = [0, 0, 12, 0, 7, 0, 12, 10]
            m = r + pat[int(round((b % 4) * 2)) % 8]
            sc.add("bgm", M.strings_short(m, (t(b + 0.5) - t(b)) * 0.8), T(b), gain, pan=-0.2)
            b += 0.5

    def chord(b0, b1, notes, gain=0.3, kind="pad"):
        d = t(b1) - t(b0) + 0.3
        if kind == "pad":
            sc.add("bgm", M.pad(notes, d, 1600, 0.3), T(b0), gain)
        else:
            sc.add("bgm", M.choir(notes, d, 0.6), T(b0), gain)

    # 章节卡：静音一拍后随重音落下（定音鼓 + 铜管大三和弦 + 低频冲击），再盖印 + 底鼓进场 + 军鼓滚奏
    sc.add("bgm", M.timpani(38, 2.4, 1.3), T(0), 0.9)
    sc.add("bgm", M.brass_stab([62, 66, 69, 74, 50], 1.2), T(0), 0.65)
    sc.add("bgm", M.cymbal(2.5), T(0), 0.2)
    sc.add("sfx", M.real("FX/FX-11/FX-11_02_cinematic_low_hit_pixabay.mp3"), T(0), 0.6)
    sc.level(T(1.5), T(4), -6, -2)                     # 重音之后的铺底别被自动电平抬满
    sc.add("sfx", M.stamp(), T(2.5), 0.7)
    sc.add("bgm", M.pad([38, 45, 50], t(8)), T(0), 0.2)
    drums(4, 7, 0.7, hats=False, snare_on=False)
    b = 7.0
    while b < 7.95:
        sc.add("bgm", M.snare(0.4 + 0.6 * (b - 7)), T(b), 0.3)
        b += 0.125
    # 4A
    sc.add("sfx", M.filt(M.real("AU/AU-01/AU-01_01_sputnik_signal_nasa.mp3", fade=0.01), "low", 3000), T(ST[1] + 0.5), 0.6)
    chord(ST[1], ST[2], [50, 57, 62], 0.25)
    drums(ST[1] + 4, ST[2], 0.5, hats=True, snare_on=False)
    chord(ST[2], ST[3], [46, 53, 58], 0.22)                   # 莱卡：鼓收住
    for k in (3, 4, 5, 6):
        drums(ST[k], ST[k + 1], 0.9)
        bass(ST[k], ST[k + 1], [38, 34, 41, 36])
    chord(ST[3], ST[7], DM + [62], 0.2)
    sc.add("sfx", M.boom(1.2, 70, 40, 0.1), T(ST[3] + 2.5), 0.3)
    sc.add("bgm", M.brass_stab([62, 65, 69, 74, 50], 0.8), T(ST[4]), 0.5)
    sc.add("bgm", M.timpani(38, 1.6, 1.1), T(ST[4]), 0.7)
    breath = M.filt(M.noise(t(ST[7]) - t(ST[6])), "band", [300, 1500]) * (0.5 + 0.5 * np.sin(
        2 * np.pi * 0.45 * np.arange(int((t(ST[7]) - t(ST[6])) * M.SR)) / M.SR)) ** 2
    sc.add("sfx", breath, T(ST[6]), 0.12)
    # 4B 地出：鼓撤掉，弦乐与合唱在颜色回来时铺开
    chord(ST[7], ST[8], [50, 57, 62, 65], 0.35)
    chord(ST[7] + 3, ST[8], [62, 65, 69, 74], 0.4, "choir")
    sc.add("bgm", M.bell(81, 3.0), T(ST[7] + 3), 0.25)
    # 阿波罗 11 号：倒数原声 → 第 2 拍点火 → 第一次大高潮
    sc.add("sfx", M.real("AU/AU-02/AU-02_01_apollo11_countdown_nasa.mp3", fade=0.2, start=11.0), T(ST[8]) - 0.2, 0.55)
    sc.add("bgm", M.riser(t(ST[8] + 2) - t(ST[8] - 2)), T(ST[8] - 2), 0.3)
    hit = T(ST[8] + 2)
    roar = M.real_loudest("FX/FX-02/FX-02_01_falcon9_pad_mics_nasa.wav", 6.0)
    x = np.arange(len(roar)) / M.SR
    sc.add("sfx", M.filt(roar, "low", 4500) * np.clip(x / 0.1, 0, 1) * np.clip((6.0 - x) / 2.0, 0, 1), hit, 0.45)
    sc.add("bgm", M.timpani(38, 2.0, 1.3), hit, 0.85)
    sc.add("bgm", M.brass_stab([62, 65, 69, 74, 50], 1.0), hit, 0.6)
    sc.add("bgm", M.cymbal(2.5), hit, 0.22)
    drums(ST[8] + 2, ST[9], 1.0)
    bass(ST[8] + 2, ST[9], [38, 34, 41, 33], 0.36)
    chord(ST[8] + 2, ST[9], [62, 65, 69, 74], 0.4, "choir")
    for o, m, d in [(2, 74, 2), (4, 77, 1), (5, 76, 1), (6, 74, 2), (8, 72, 1), (9, 74, 1)]:
        sc.add("bgm", M.horn(m, t(ST[8] + o + d) - t(ST[8] + o)), T(ST[8] + o), 0.35)
    # 纽约时报更正
    sc.add("sfx", M.real("FX/FX-13/FX-13_02_printing_press_rhythm_freesound.flac", t(ST[10]) - t(ST[9]), 0.1), T(ST[9]), 0.4)
    sc.add("bgm", M.brass_stab([62, 66, 69, 74], 0.6), T(ST[9]), 0.5)
    sc.add("bgm", M.brass_stab([62, 66, 69, 74], 0.6), T(ST[9] + 2), 0.45)
    # 登月：只剩弦乐长音，Quindar 提示音 + 阿姆斯特朗原声
    chord(ST[10], ST[11], [50, 57, 62, 66], 0.3)
    sc.add("bgm", M.strings_long(74, t(ST[11]) - t(ST[10])), T(ST[10]), 0.2)
    q = M.real("FX/FX-16/FX-16_01_quindar_tones_freesound.wav", fade=0.005)
    sc.add("sfx", q, T(ST[10] + 5.4), 0.3)
    sc.add("sfx", M.real("AU/AU-02/AU-02_02_armstrong_small_step_nasa.mp3", fade=0.05), T(ST[10] + 6.0), 0.9)
    sc.add("sfx", q, T(ST[11] - 1.0), 0.25)
    # 4C 1970 年 4 月：《东方红》旋律（钟声，像卫星播放的电子乐音）
    chord(ST[11], ST[12], [43, 50, 55, 59], 0.25)
    bass(ST[11], ST[12], [31, 31], 0.25)
    mel = [(0, 67, 1), (1, 67, 0.5), (1.5, 69, 0.5), (2, 62, 2), (4, 60, 1), (5, 60, 0.5), (5.5, 57, 0.5), (6, 62, 2)]
    for o, m, d in mel:
        sc.add("bgm", M.bell(m + 12, 2.0), T(ST[11] + 1 + o), 0.35)
    chord(ST[12], ST[13], [45, 52, 57], 0.22)                 # 联盟 11 号
    chord(ST[13], ST[14], [50, 57, 62], 0.25)
    for k, m in enumerate([74, 72, 69, 67, 65, 62]):
        sc.add("bgm", M.strings_long(m, t(ST[13] + 2 * k + 2) - t(ST[13] + 2 * k)), T(ST[13] + 2 * k), 0.18)
    sc.add("sfx", M.real("FX/FX-07/FX-07_01_soft_wind_pixabay.mp3", t(ST[14]) - t(ST[13] + 8), 0.5), T(ST[13] + 8), 0.25)
    # 4D 旅行者：钟声琶音 + 轻鼓
    chord(ST[14], ST[15], [50, 57, 62, 69], 0.3)
    for i in range(24):
        m = [62, 69, 74, 77, 81, 77, 74, 69][i % 8]
        sc.add("bgm", M.bell(m, 1.6), T(ST[14] + i * 0.5), 0.16)
    drums(ST[14] + 4, ST[15], 0.5, snare_on=False)
    # 航天飞机
    roar2 = M.real_loudest("FX/FX-02/FX-02_01_falcon9_pad_mics_nasa.wav", 3.5)
    x = np.arange(len(roar2)) / M.SR
    sc.add("sfx", M.filt(roar2, "low", 4500) * np.clip((3.5 - x) / 1.5, 0, 1), T(ST[15]), 0.4)
    sc.add("bgm", M.timpani(38, 1.6, 1.1), T(ST[15]), 0.7)
    drums(ST[15], ST[16], 0.9)
    bass(ST[15], ST[16], [38, 41], 0.3)
    chord(ST[15], ST[16], [62, 65, 69], 0.3, "choir")
    # 挑战者号：没有鼓，只有低音铺底
    chord(ST[16], ST[17], [38, 45, 50], 0.22)
    # 暗淡蓝点：完全静音，最后落下一个音
    sc.silence(T(ST[17]) - 0.02, T(ST[18]) - t(ST[18]) + t(ST[18] - 0.6))
    sc.add("bgm", M.bell(62, 4.0), T(ST[18] - 0.5), 0.35)
    # 哈勃：音乐重新起
    chord(ST[18], ST[19], [50, 57, 62, 65], 0.3)
    for i in range(12):
        sc.add("bgm", M.pluck([62, 65, 69, 74][i % 4], 1.2, 1.1, 0.6), T(ST[18] + i * 0.5), 0.25)
    # 年份变化时的电传声（暗淡蓝点、纪念镜头除外）
    tty = M.real("FX/FX-12/FX-12_01_vintage_teletype_freesound.wav", 0.35, 0.02)
    for k in range(1, 19):
        if k in (12, 16, 17):
            continue
        sc.add("sfx", tty, T(ST[k]), 0.16)


CHAPTER = Chapter("race_a", "竞赛", 120, SHOTS, overlay=overlay, clips=[A11], music=music)
