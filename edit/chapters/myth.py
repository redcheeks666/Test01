"""壹 · 神话（72 BPM）。嫦娥、飞天、天问三段沿用样片画面；阿尔忒弥斯用大都会博物馆陶瓶实物重做。"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from engine.core import (W, H, FPS, CREAM, INK, GOLD, ROOT, chapter_card, post, place, lerp, ease_io, ease_out,
                         ease_out_expo, clamp01, fbm, glow, caption, circle_mask, xx, yy)
from engine.film import Chapter, Shot
from engine.media import Clip, image
from engine import music as M

SAMPLE = ROOT / "reference" / "sample_v0.mp4"
CHANGE = Clip("smp_change", SAMPLE, 32.0, 42.0, size=(1920, 1080))
FEITIAN = Clip("smp_feitian", SAMPLE, 42.0, 52.8, size=(1920, 1080))
TIANWEN = Clip("smp_tianwen", SAMPLE, 63.8, 82.0, size=(1920, 1080))


def s_card(c):
    return chapter_card(c, "神话")


def _sample(c, clip, src_dur):
    f = clip.at(c.lsec * src_dur / c.dur, blend=False).copy()
    return post(f, c.fi, grain=0.015, vignette=False, weave=False)


def s_change(c):
    return _sample(c, CHANGE, 10.0)


def s_feitian(c):
    return _sample(c, FEITIAN, 10.8)


def s_tianwen(c):
    f = _sample(c, TIANWEN, 18.2)
    return f * (1 - clamp01((c.lb - (c.beats - 1.0)) / 1.0))


# ---------------- 阿尔忒弥斯 ----------------

def _greek_bg():
    cx, cy = 1180, 520
    d = np.sqrt(((xx - cx) / W) ** 2 + ((yy - cy) / H) ** 2)
    k = np.clip(d / 0.75, 0, 1)[..., None]
    inner, outer = np.array([0.86, 0.53, 0.30], np.float32), np.array([0.45, 0.22, 0.12], np.float32)
    bg = inner * (1 - k) + outer * k
    bg = bg * (0.9 + 0.16 * fbm(H, W, 6, 51)[..., None])
    # 上下回纹带
    band = Image.new("RGBA", (W, 70), (56, 30, 18, 255))
    d_ = ImageDraw.Draw(band)
    col = (232, 206, 160, 255)
    u = 50
    for x0 in range(-u, W + u, u):
        pts = [(x0, 55), (x0, 15), (x0 + 36, 15), (x0 + 36, 43), (x0 + 12, 43), (x0 + 12, 27), (x0 + 24, 27)]
        d_.line(pts, fill=col, width=4)
        d_.line([(x0, 55), (x0 + u, 55)], fill=col, width=4)
    d_.line([(0, 6), (W, 6)], fill=col, width=3)
    d_.line([(0, 64), (W, 64)], fill=col, width=3)
    b = np.asarray(band, np.float32)[..., :3] / 255
    bg[:70] = b
    bg[H - 70:] = b[::-1]
    return np.clip(bg, 0, 1).astype(np.float32)


GREEK = _greek_bg()


def _medallion(frame, src, cx_n, cy_n, scale, sx, sy, r, ring_k=1.0, alpha=1.0):
    rgb, a = place(src, cx_n * src.shape[1], cy_n * src.shape[0], scale, 0, sx, sy)
    m = circle_mask(sx, sy, r, 2.0) * a * alpha
    frame = frame * (1 - m * 0.0) * (1 - circle_mask(sx + 12, sy + 16, r + 4, 18) * 0.35 * alpha)  # 投影
    frame = frame * (1 - m) + rgb * m
    if ring_k > 0:
        ang = np.degrees(np.arctan2(yy - sy, xx - sx)) + 90
        ang = np.where(ang < 0, ang + 360, ang)
        d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        ring = (np.clip(1 - np.abs(d - r - 8) / 5, 0, 1) * (ang <= ring_k * 360))[..., None] * alpha
        frame = frame * (1 - ring) + np.array(GOLD, np.float32) * ring
    return frame


def _crescent(frame, x, y, r, alpha):
    m = (circle_mask(x, y, r, 1.5) * (1 - circle_mask(x + r * 0.42, y - r * 0.18, r * 0.92, 1.5)))
    frame = frame + glow(x, y, r * 1.6, (0.5, 0.4, 0.2), 0.35 * alpha)
    return frame * (1 - m * alpha) + np.array([0.98, 0.9, 0.7], np.float32) * m * alpha


def _sun(frame, x, y, r, alpha, rot):
    frame = frame + glow(x, y, r * 1.8, (0.6, 0.4, 0.15), 0.4 * alpha)
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    ang = np.arctan2(yy - y, xx - x) + rot
    rays = ((np.cos(ang * 12) > 0.55) & (d > r * 1.15) & (d < r * 1.6))[..., None] * alpha
    disc = circle_mask(x, y, r, 1.5) * alpha
    col = np.array([0.98, 0.84, 0.5], np.float32)
    frame = frame * (1 - rays) + col * rays
    return frame * (1 - disc) + col * disc


def s_artemis(c):
    f = GREEK.copy()
    lb = c.lb
    art = image("PD-13", "01", 2400)
    apo = image("PD-13", "02", 2400)
    twin = ease_io(clamp01((lb - 7.0) / 1.2))
    # 阿尔忒弥斯徽章：先居中，第 7 拍起移到左侧，让出位置给阿波罗
    r1 = lerp(300, 255, twin)
    sx1 = lerp(1250, 1060, twin)
    s1 = lerp(0.62, 0.66, ease_io(c.k)) * r1 / 300
    f = _medallion(f, art, 0.52, 0.57, s1, sx1, 580, r1, ring_k=ease_out(clamp01((lb - 0.2) / 1.6), 2),
                   alpha=clamp01(c.lsec / 0.6))
    f = _crescent(f, sx1, 580 - r1 - 95, 52, clamp01((lb - 1.8) * 1.5))
    if twin > 0:
        r2 = 255
        sx2 = lerp(1900, 1610, twin)
        f = _medallion(f, apo, 0.33, 0.50, 0.52 * r2 / 255, sx2, 580, r2, ring_k=twin, alpha=twin)
        f = _sun(f, sx2, 580 - r2 - 95, 44, twin, c.lsec * 0.3)
    caption(f, c, 0.6, [("同一个月亮，在地中海的另一端", "lead"), ("阿尔忒弥斯", "big"),
                        ("希腊神话里的月亮女神", "mid")], y=120, scrim=0.35)
    caption(f, c, 7.6, [("太阳神阿波罗的孪生姐姐", "mid")], y=120 + 52 + 158 + 70, scrim=0.0)
    return post(f, c.fi, grain=0.02, vig_amt=0.45)


SHOTS = [Shot(8, s_card, name="1-01"), Shot(12, s_change, name="1-02"), Shot(13, s_feitian, name="1-03"),
         Shot(14, s_artemis, name="1-04"), Shot(22, s_tianwen, name="1-05")]

# D 小调五声：D F G A C
PENTA = [50, 53, 55, 57, 60, 62, 65, 67, 69, 72, 74]


def music(sc, ch):
    T, t = ch.T, ch.t
    total = ch.duration
    # 章节卡：纸张翻动 + 盖印
    sc.add("sfx", M.real("FX/FX-05/FX-05_01_paper_turn_pixabay.mp3", fade=0.02), T(0.2), 0.3)
    sc.add("sfx", M.stamp(), T(2.5), 0.6)
    # 全章铺底与风声
    sc.add("bgm", M.pad([38, 45, 50, 57], total, 900, 2.0), T(0), 0.35)
    sc.add("sfx", M.real("FX/FX-07/FX-07_01_soft_wind_pixabay.mp3", t(33) - t(8), 1.0), T(8), 0.25)
    # 古琴：八分音符分解，每拍一组，句尾留白
    r = np.random.default_rng(3)
    for b in np.arange(8, ch.total_beats - 2, 0.5):
        if (b % 8) >= 7:
            continue
        if r.random() < 0.72:
            m = PENTA[int(r.integers(0, 7))] - 12 if b % 2 == 0 else PENTA[int(r.integers(2, 9))] - 12
            sc.add("bgm", M.pluck(m, 2.2, 0.8, 1.2, slide=0.01), T(b), 0.32, pan=r.uniform(-0.3, 0.3))
    # 箫：几段旋律
    phrases = [(9, [(62, 2), (65, 1), (67, 1), (69, 3), (67, 1)]),
               (21, [(69, 2), (72, 1), (74, 1), (72, 2), (69, 2)]),
               (34, [(74, 2), (72, 1), (69, 1), (67, 2), (65, 2)]),
               (47, [(62, 1.5), (65, 0.5), (67, 2), (69, 2), (72, 2)]),
               (59, [(74, 3), (72, 1), (69, 2), (67, 2)]),
               (69, [(65, 2), (67, 2), (62, 4)])]
    for b0, notes in phrases:
        b = b0
        for m, d in notes:
            sc.add("bgm", M.xiao(m, t(b + d) - t(b)), T(b), 0.42, pan=0.1)
            b += d
    # 飞天：飘带掠过
    for b in (34, 38, 42):
        sc.add("sfx", M.whoosh(0.9, 300, 2500), T(b), 0.12)
    # 阿尔忒弥斯：竖琴般的上行拨弦 + 第 7 拍孪生揭示的钟声
    b_art = ch.shot_start(3)
    for k, m in enumerate([62, 65, 69, 74, 77, 81]):
        sc.add("bgm", M.pluck(m, 2.0, 1.2, 1.0), T(b_art + 0.2 + k * 0.25), 0.25)
    sc.add("bgm", M.bell(74, 3.0), T(b_art + 7.0), 0.3)
    sc.add("bgm", M.bell(81, 3.0), T(b_art + 7.5), 0.2)
    # 天问：竹简碰撞（纸声 + 木声），问号飞升化作星辰时的钟声
    b_tw = ch.shot_start(4)
    for k in range(6):
        sc.add("sfx", M.tick() * 3, T(b_tw + 0.4 + k * 0.35), 0.25)
    sc.add("sfx", M.real("FX/FX-05/FX-05_01_paper_turn_pixabay.mp3", fade=0.02), T(b_tw + 0.3), 0.35)
    for k, m in enumerate([74, 77, 81, 86, 89]):
        sc.add("bgm", M.bell(m, 2.5), T(b_tw + 11 + k * 0.5), 0.18)


CHAPTER = Chapter("myth", "神话", 72, SHOTS, clips=[CHANGE, FEITIAN, TIANWEN], music=music)
