"""尾声（60 BPM）。47 → 33 → 旅行者 1 号的一光日 → 结语 → 月亮与片名 → 片尾署名。"""
import cv2
import numpy as np

from engine.core import (W, H, FPS, CREAM, INK, GOLD, PAPER, chapter_card, post, place, lerp, ease_io, ease_in,
                         ease_out, ease_out_expo, clamp01, glow, caption, grade_clean, draw_text, starfield, SEAL,
                         text_width, duotone)
from engine.film import Chapter, Shot
from engine.media import image
from engine.shots import ken
from engine import music as M
from chapters import prologue as P

STARS = starfield(41, 900)


def _ring_rockets(n=47, r_out=330):
    """47 支火箭排成同心圆阵（线描），为匹配剪辑成 33 台发动机做准备。"""
    f = np.zeros((H, W, 3), np.float32) + np.array([0.05, 0.04, 0.04], np.float32)
    rings = [(1, 0), (6, 90), (14, 190), (26, 300)]          # 共 47
    pts = []
    for cnt, r in rings:
        for i in range(cnt):
            a = 2 * np.pi * i / cnt + (0.2 if r else 0)
            pts.append((W / 2 + 120 + r * np.cos(a), H / 2 + r * np.sin(a)))
    return f, pts


RING_BG, RING_PTS = _ring_rockets()


def s_47(c):
    k = ease_io(clamp01((c.lb - 3.0) / 1.5))
    f = RING_BG.copy()
    n = min(47, int(47 * clamp01(c.lb / 2.2)))
    for x, y in RING_PTS[:n]:
        cv2.circle(f, (int(x), int(y)), 26, (0.85, 0.72, 0.45), 2, cv2.LINE_AA)
        cv2.circle(f, (int(x), int(y)), 9, (0.95, 0.6, 0.25), -1, cv2.LINE_AA)
    if k > 0:
        ph = grade_clean(image("SX-06", "01", 2400), 0.0, 1.05, 1.05)
        p = ken(c, ph, (0.47, 0.42, 1.25), (0.47, 0.40, 1.12), t0=3, t1=8)
        f = f * (1 - k) + p * k
    big = "47" if c.lb < 3.4 else "33"
    draw_text(f, big, "mono", 150, CREAM, 300, H / 2 - 40, 1.0, anchor="c")
    if c.lb >= 4.5:
        caption(f, c, 4.5, [("明朝的 47 支火箭", "mid"), ("今天的 33 台猛禽发动机", "mid")], y=H - 300)
    sh, e = (0, 0), 0
    return post(f, c.fi, flash=0.0, grain=0.03)


def s_voyager(c):
    if c.lb < 3:
        src = duotone(image("NA-18", "01", 2400), (0.06, 0.06, 0.08), (0.9, 0.9, 0.86))
        f = ken(c, src, (0.6, 0.45, 1.05), (0.62, 0.42, 1.15), t0=0, t1=3)
        f = f * (1 - clamp01((c.lb - 2.4) / 0.6)) + STARS * clamp01((c.lb - 2.4) / 0.6)
    else:
        f = STARS.copy()
        f = f + glow(W / 2, H / 2 - 40, 6, (1, 1, 1), 0.9)
        km = 25_600_000_000 + int(300_000_000 * clamp01((c.lb - 3) / 3))
        num = f"{km:,}"
        nw, uw, gap = text_width(num, "mono", 64, 4), text_width("公里", "serif", 40), 18
        x0 = W / 2 - (nw + gap + uw) / 2                      # 数字 + 单位整体居中
        draw_text(f, num, "mono", 64, CREAM, x0 + nw / 2, H / 2 + 90, 1.0, anchor="c", tracking=4)
        draw_text(f, "公里", "serif", 40, CREAM, x0 + nw + gap + uw / 2, H / 2 + 94, 1.0, anchor="c")
        draw_text(f, "距离地球", "serif", 34, (0.8, 0.78, 0.72), W / 2, H / 2 + 175, 1.0, anchor="c")
    caption(f, c, 0.4, [("旅行者 1 号", "big"), ("2026 年 11 月，它将距离地球一光日", "mid")])
    return post(f, c.fi, grain=0.025)


def s_words(c):
    f = STARS.copy()
    a1 = clamp01((c.lb - 0.5) / 1.0)
    a2 = clamp01((c.lb - 3.5) / 1.0)
    out = 1 - clamp01((c.lb - (c.beats - 1)) / 1.0)
    draw_text(f, "五千年前，我们把梦写进神话", "serif", 54, CREAM, W / 2, H / 2 - 60, a1 * out, anchor="c", tracking=10)
    draw_text(f, "今天，我们把它送上轨道", "serif", 54, CREAM, W / 2, H / 2 + 50, a2 * out, anchor="c", tracking=10)
    return post(f, c.fi, grain=0.02, vignette=False)


def s_title(c):
    rise = ease_out(clamp01(c.lb / 3.0), 3)
    f = P.moon_layer(lerp(H + 380, H / 2 - 20, rise), 330, STARS.copy())
    if c.lb >= 3.0:
        k = clamp01(c.since(3.0) * FPS / 14)
        draw_text(f, "天问", "brush", 230, INK, W / 2, H / 2 - 30, k * 0.92, lerp(1.08, 1, ease_out_expo(k)),
                  tracking=40, anchor="c", shadow=False)
        if c.lb >= 3.6:
            fr = c.since(3.6) * FPS
            ss = lerp(1.5, 1.0, ease_out_expo(fr / 5))
            rgb, a = SEAL
            s2 = int(80 * ss)
            rs, as_ = cv2.resize(rgb, (s2, s2)), cv2.resize(a, (s2, s2))[..., None]
            x0, y0 = int(W / 2 + 230 - s2 / 2), int(H / 2 + 40 - s2 / 2)
            kk = min(fr / 2, 1)
            f[y0:y0 + s2, x0:x0 + s2] = f[y0:y0 + s2, x0:x0 + s2] * (1 - as_ * kk) + rs * as_ * kk
    out = 1 - clamp01((c.lb - (c.beats - 1.2)) / 1.2)
    return post(f * out, c.fi, grain=0.02, vignette=False)


CREDITS = [
    ("天问", "title"),
    ("人类向往星空的五千年", "sub"),
    ("", "gap"),
    ("音乐", "head"),
    ("配乐为程序合成", "line"),
    ("", "gap"),
    ("音效", "head"),
    ("取自真实的航天录音与 CC0 音效素材", "line"),
    ("", "gap"),
    ("说明", "head"),
    ("万户的故事为传说", "line"),
    ("万户、斯普特尼克 1 号、莱卡、加加林、列昂诺夫、神舟五号、月船 3 号、", "line"),
    ("凯利滑翔机、皮卡尔气球、张衡浑天仪等画面为代码绘制的示意", "line"),
    ("报纸版面为重绘示意 · 神话段落沿用样片中的绘制画面", "line"),
    ("", "gap"),
    ("", "gap"),
    ("谨以此片，致敬每一位为飞向星空付出过的人", "sub"),
]


def s_credits(c):
    f = STARS * 0.5
    spec = {"title": ("brush", 110, 170), "sub": ("serif", 36, 76), "head": ("serif", 34, 66),
            "line": ("serif", 30, 50), "gap": ("serif", 20, 44)}
    last = sum(spec[k][2] for _, k in CREDITS[:-1])           # 最后一行（致敬）相对首行的偏移
    y0 = lerp(H + 40, H / 2 - last, ease_out(clamp01(c.lb / (c.beats - 4)), 1.6))   # 致敬停在画面中央，留 4 拍
    y = y0
    for text, kind in CREDITS:
        font_k, size, step = spec[kind]
        col = GOLD if kind == "head" else CREAM
        if -100 < y < H + 100 and text:
            draw_text(f, text, font_k, size, col, W / 2, y, 1.0, anchor="c", tracking=4 if kind != "line" else 1)
        y += step
    out = 1 - clamp01((c.lb - (c.beats - 1.5)) / 1.5)
    return post(f * out, c.fi, grain=0.015, vignette=False)


SHOTS = [Shot(8, s_47, name="E-01"), Shot(7, s_voyager, name="E-02"), Shot(7, s_words, name="E-03"),
         Shot(9, s_title, name="E-04"), Shot(20, s_credits, name="E-05")]
ST = np.concatenate([[0], np.cumsum([s.beats for s in SHOTS])])


def music(sc, ch):
    T, t = ch.T, ch.t
    sc.add("sfx", M.real("FX/FX-11/FX-11_02_cinematic_low_hit_pixabay.mp3"), T(3.0), 0.45)
    sc.add("bgm", M.pad([38, 45, 50, 57], ch.duration, 1000, 1.5), T(0), 0.35)
    sc.add("bgm", M.timpani(38, 2.0, 1.0), T(3.0), 0.6)
    sc.add("bgm", M.choir([62, 66, 69], t(8) - t(3), 0.5), T(3.0), 0.35)
    sc.add("sfx", M.real("FX/FX-09/FX-09_01_radio_static_pixabay.mp3", t(ST[2]) - t(ST[1] + 3), 0.3), T(ST[1] + 3), 0.15)
    # 箫：主题回归
    phrases = [(ST[2] + 0.5, [(62, 1.5), (65, 0.5), (67, 1), (69, 2)]), (ST[2] + 4, [(72, 1), (69, 1), (67, 1)]),
               (ST[3] + 3.5, [(62, 1.5), (65, 0.8), (67, 0.8), (69, 3)])]
    for b0, notes in phrases:
        b = b0
        for m, d in notes:
            sc.add("bgm", M.xiao(m, t(b + d) - t(b)), T(b), 0.5)
            b += d
    sc.add("bgm", M.gong(8.0), T(ST[3] + 3.0), 0.9)
    # 片尾：古琴
    r = np.random.default_rng(9)
    pent = [50, 53, 55, 57, 60, 62]
    for b in np.arange(ST[4], ST[5] - 2, 1.0):
        sc.add("bgm", M.pluck(pent[int(r.integers(0, 6))], 3.0, 0.8, 1.4), T(b), 0.3, pan=r.uniform(-0.3, 0.3))
        if r.random() < 0.5:
            sc.add("bgm", M.pluck(pent[int(r.integers(0, 6))] + 12, 2.0, 0.9, 1.0), T(b + 0.5), 0.2)


CHAPTER = Chapter("epilogue", "尾声", 60, SHOTS, music=music)
