"""肆 · 竞赛（后半，128 BPM）：2000—2024。国际空间站 → 哥伦比亚号 → 神舟五号（水墨）→
加速快切：星际空间、冥王星、嫦娥四号、2021 年春天（机智号 | 祝融号，"天问"二字飞落）、韦布、天宫、阿尔忒弥斯 1 号、月船 3 号、嫦娥六号。"""
import cv2
import numpy as np

from engine.core import (W, H, FPS, CREAM, INK, GOLD, PAPER, PAPER_DARK, SMOKE, post, place, lerp, ease_io, ease_in,
                         ease_out, ease_out_expo, clamp01, glow, caption, grade_clean, draw_text, starfield,
                         circle_mask, flash_at, shake_at, fbm, xx, yy, YearTicker, cover_scale, fit_scale, motion_blur)
from engine.film import Chapter, Shot
from engine.media import image
from engine.shots import ken, split, memorial
from engine import music as M
from chapters import race_art as A
from chapters.race_a import ST as ST_A, TICKER as TICKER_A

SPACE = starfield(33, 700, (0.01, 0.012, 0.02))
_c = {}


def img(aid, cand="01", crop=None, max_edge=3000):
    key = (aid, cand, crop)
    if key not in _c:
        x = image(aid, cand, max_edge)
        if crop:
            h, w = x.shape[:2]
            x = x[int(crop[1] * h):int(crop[3] * h), int(crop[0] * w):int(crop[2] * w)]
        _c[key] = grade_clean(x, 0.2, 1.06, 1.05)
    return _c[key]


def hud(f, k=1.0, label=""):
    """细线取景框：四角括号 + 小标签。"""
    col = np.array([0.85, 0.9, 0.95], np.float32)
    L, m, t_ = 60, 60, 2
    for (x, y, dx, dy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)):
        x0, x1 = sorted((x, x + dx * int(L * k)))
        y0, y1 = sorted((y, y + dy * int(L * k)))
        f[y - (t_ if dy > 0 else 0):y + (t_ if dy > 0 else 0) or None, x0:x1] = col
        f[y0:y1, x - (t_ if dx > 0 else 0):x + (t_ if dx > 0 else 0) or None] = col
    if label:
        fk = "mono" if label.isascii() else "serif"
        draw_text(f, label, fk, 20, (0.85, 0.9, 0.95), W - 80, 72, k, anchor="rt", shadow=True)
    return f


def ink_reveal(k, seed=0):
    thr = lerp(1.1, -0.1, ease_io(k))
    return np.clip((SMOKE[seed % 2] - thr) * 5, 0, 1)[..., None]


def ink_card(c, src_img, cx, cy, width, k_in, push=(1.0, 1.06), rot=(0, 0)):
    """宣纸上的照片：水墨晕染入画。"""
    f = PAPER.copy()
    rgb, a = place(src_img, src_img.shape[1] / 2, src_img.shape[0] / 2,
                   width / src_img.shape[1] * lerp(push[0], push[1], c.k), lerp(rot[0], rot[1], c.k), cx, cy)
    rev = ink_reveal(k_in, 1)
    return f * (1 - a * rev) + rgb * a * rev


# ---------------- 镜头 ----------------

def s_iss(c):
    f = ken(c, img("NA-25"), (0.30, 0.55, 1.15), (0.70, 0.45, 1.25))
    f = hud(f, ease_out(clamp01(c.lb), 2), "ISS")
    caption(f, c, 0.4, [("国际空间站", "big"), ("从这一天起，太空中一直有人", "mid")])
    return post(f, c.fi, grain=0.03)


def s_columbia(c):
    return memorial(c, image("NA-26", "01", 1600), "2003 · 哥伦比亚号",
                    ["里克·赫斯本德", "威廉·麦库尔", "迈克尔·安德森", "卡尔帕纳·乔拉", "戴维·布朗", "劳雷尔·克拉克",
                     "伊兰·拉蒙"], sub="返回时解体，七名航天员遇难", hold=4.0)


LM = A.long_march(860)


def s_shenzhou(c):
    f = PAPER.copy()
    rise = ease_in(clamp01((c.lb - 1.5) / 6.5), 2.0)
    cx, cy = 1300, lerp(620, 120, rise)
    A.paste(f, *LM, cx, cy, 1.0)
    # 墨色尾烟：从火箭底部向两侧晕开
    base_y = cy + LM[1].shape[0] / 2
    n = SMOKE[0] * 0.6 + SMOKE[1] * 0.4
    d = np.sqrt(((xx - cx) / 1.6) ** 2 + (np.clip(yy - base_y, 0, None)) ** 2)
    grow = lerp(80, 520, ease_out(clamp01((c.lb - 1.0) / 5), 2))
    smoke = np.clip((grow - d) / 140 + (n - 0.5) * 1.6, 0, 1) * (yy > base_y - 30)
    f = f * (1 - smoke[..., None] * 0.92) + np.array([0.08, 0.07, 0.06]) * smoke[..., None] * 0.92
    flame = glow(cx, base_y + 30, 30, (1.0, 0.5, 0.15), 1.2 * clamp01((c.lb - 1.2) * 2), sx=26, sy=60)
    f = f + flame
    f = PAPER * (1 - ink_reveal(clamp01(c.lb / 1.2))) + f * ink_reveal(clamp01(c.lb / 1.2))
    caption(f, c, 0.6, [("神舟五号", "big"), ("杨利伟进入太空", "mid"),
                        ("中国成为第三个独立掌握载人航天技术的国家", "small")], color=INK, scrim=0.0)
    sh, _ = shake_at(c, 1.5, 6, 0.6)
    return post(f, c.fi, grain=0.025, shake=sh)


def s_interstellar(c):
    f = ken(c, img("NA-27"), (0.42, 0.5, 1.15), (0.62, 0.5, 1.3))
    f = hud(f, 1.0, "NASA/JPL · 示意图")
    caption(f, c, 0.2, [("星际空间", "big"), ("旅行者 1 号飞出了日球层", "mid")])
    return post(f, c.fi, grain=0.025)


def s_pluto(c):
    src = img("NA-28")
    z = lerp(2.2, 1.0, ease_out_expo(c.since(0) * FPS / 10)) * lerp(1.0, 1.05, c.k)
    rgb, a = place(src, src.shape[1] / 2, src.shape[0] / 2, 820 / src.shape[0] * z, 0, 1250, 540)
    f = SPACE * (1 - a) + rgb * a
    f = hud(f, 1.0, "NEW HORIZONS")
    caption(f, c, 0.2, [("冥王星", "big"), ("新视野号飞掠", "mid")])
    return post(f, c.fi, grain=0.025)


def s_change4(c):
    f = ink_card(c, image("CN-03", "01", 1400), 1250, 560, 1000, clamp01(c.lb / 1.0))
    caption(f, c, 0.3, [("嫦娥四号", "big"), ("人类第一次在月球背面着陆", "mid")], color=INK, scrim=0.0)
    return post(f, c.fi, grain=0.025)


TIANWEN_CHARS = "天问"


def s_spring2021(c):
    left = np.zeros((H, W, 3), np.float32)
    ing = img("NA-29")
    l_ = ken(c, ing, (0.5, 0.5, 1.0), (0.5, 0.5, 1.08), out_w=W // 2)
    left[:, : W // 2] = l_
    left = hud(left, 1.0, "") if False else left
    right = ink_card(c, image("CN-04", "02", 1920), W * 0.75, H * 0.45, 820, clamp01((c.lb - 0.3) / 1.0))
    rh = np.zeros_like(right)
    rh[:, : W // 2] = right[:, W // 2:]
    f = split(left, rh, line=ease_out_expo(c.since(0) * FPS / 8))
    # "天问"二字从左上沿弧线飞到祝融号上方
    k = ease_io(clamp01((c.lb - 2.5) / 2.0))
    if c.lb >= 2.5:
        x = lerp(W * 0.30, W * 0.75, k)
        y = lerp(H * 0.20, H * 0.25, k) - np.sin(k * np.pi) * 140
        s = lerp(1.6, 1.0, k)
        draw_text(f, TIANWEN_CHARS, "brush", int(96 * s), INK if k > 0.5 else CREAM, x, y, 1.0, anchor="c", tracking=10,
                  shadow=k <= 0.5)
    if c.lb >= 4.5:
        f = f + glow(W * 0.75, H * 0.25, 120, (0.3, 0.2, 0.05), 0.4 * (1 - clamp01((c.lb - 4.5) / 1.5)))
    fr = c.since(0) * FPS
    draw_text(f, "2021 年春天", "brush", 84, CREAM, W * 0.25, 110, min(fr / 4, 1), lerp(1.25, 1, ease_out_expo(fr / 9)),
              anchor="c")
    a2 = clamp01((c.lb - 0.8) * 2)
    f[int(H * 0.66):int(H * 0.77), : W // 2] *= 0.45
    draw_text(f, "4.19 · 机智号：第一次在另一颗行星上飞行", "serif", 30, CREAM, W * 0.25, H * 0.715, a2, anchor="c")
    draw_text(f, "5.15 · 天问一号：祝融号登陆火星", "serif", 30, INK, W * 0.75, H * 0.80, a2, anchor="c", shadow=False)
    return post(f, c.fi, grain=0.025)


def s_jwst(c):
    src = img("NA-30")
    z = 1.0 * (2.4 ** ease_in(c.k, 1.4))
    f = ken(c, src, (0.5, 0.5, z), (0.5, 0.5, z))
    f = motion_blur(f, 3, 0)
    f = hud(f, 1.0, "JWST · NIRCam")
    caption(f, c, 0.3, [("韦布空间望远镜", "big"), ("第一张深场图", "mid")])
    return post(f, c.fi, grain=0.02)


def s_tiangong(c):
    f = ink_card(c, image("CN-05", "01", 2400)[int(0.13 * 1728):int(0.86 * 1728)], 1240, 560, 1150,
                 clamp01(c.lb / 0.8))
    caption(f, c, 0.2, [("天宫空间站", "big"), ("建成", "mid")], color=INK, scrim=0.0)
    return post(f, c.fi, grain=0.025)


def s_artemis1(c):
    f = ken(c, img("NA-31"), (0.55, 0.5, 1.1), (0.5, 0.45, 1.22))
    f = hud(f, 1.0, "ORION · ARTEMIS I")
    caption(f, c, 0.2, [("Artemis I", "big"), ("无人绕月试飞", "mid")])
    return post(f, c.fi, grain=0.025)


LANDER = A.lander(360)


def s_chandrayaan(c):
    f = SPACE.copy()
    ground = (yy > 760 + 60 * np.sin(xx / 300)).astype(np.float32)[..., None]
    gtex = (0.32 + 0.12 * fbm(H, W, 6, 91))[..., None]
    f = f * (1 - ground) + gtex * ground
    f = f + glow(1600, 220, 40, (0.4, 0.5, 0.8), 0.8)
    f = f * (1 - circle_mask(1600, 220, 34, 1.5)) + np.array([0.35, 0.5, 0.85]) * circle_mask(1600, 220, 34, 1.5)
    A.paste(f, *LANDER, 1250, lerp(690, 700, c.k), 1.0)
    f = hud(f, 1.0, "")
    caption(f, c, 0.2, [("月船 3 号", "big"), ("印度 · 在月球南极附近着陆", "mid")])
    return post(f, c.fi, grain=0.025)


def s_change6(c):
    if c.lb < 3:
        f = ink_card(c, image("CN-06", "01", 1400), 1250, 560, 1000, clamp01(c.lb / 0.8))
    else:
        f = ink_card(c, image("CN-06", "02", 1400), 1250, 560, 1000, clamp01((c.lb - 3) / 0.8))
    caption(f, c, 0.3, [("嫦娥六号", "big"), ("第一次带回月球背面的样本", "mid")], color=INK, scrim=0.0)
    return post(f * (1 - clamp01((c.lb - (c.beats - 0.5)) * 2)), c.fi, grain=0.025)


SHOTS = [Shot(6, s_iss, name="4-20"), Shot(10, s_columbia, name="4-21"), Shot(8, s_shenzhou, name="4-22"),
         Shot(3, s_interstellar, name="4-23"), Shot(3, s_pluto, name="4-24"), Shot(4, s_change4, name="4-25"),
         Shot(8, s_spring2021, name="4-26"), Shot(4, s_jwst, name="4-27"), Shot(3, s_tiangong, name="4-28"),
         Shot(3, s_artemis1, name="4-29"), Shot(3, s_chandrayaan, name="4-30"), Shot(6, s_change6, name="4-31")]
ST = np.concatenate([[0], np.cumsum([s.beats for s in SHOTS])])

TICKER = YearTicker([(-1, 1990, "04.24"), (ST[0], 2000, "11.02"), (ST[1], 2003, "02.01"), (ST[2], 2003, "10.15"),
                     (ST[3], 2012, "08.25"), (ST[4], 2015, "07.14"), (ST[5], 2019, "01.03"), (ST[6], 2021, "春"),
                     (ST[7], 2021, "12.25"), (ST[8], 2022, ""), (ST[9], 2022, "11.16"), (ST[10], 2023, "08.23"),
                     (ST[11], 2024, "06.25")], roll_beats=1.4)


def overlay(img_, c):
    a = 1 - clamp01((c.b - (ST[-1] - 0.5)) * 2)
    return TICKER.draw(img_, c.b, a)


DMP = [50, 53, 55, 57, 60, 62, 65, 67, 69, 72, 74]


def music(sc, ch):
    T, t = ch.T, ch.t

    def drums(b0, b1, gain=1.0, taiko=True):
        b = b0
        while b < b1 - 1e-6:
            if b % 1 == 0:
                sc.add("bgm", M.kick(1.0), T(b), 0.55 * gain)
            if b % 2 == 1:
                sc.add("bgm", M.snare(0.9), T(b), 0.3 * gain)
            sc.add("bgm", M.hat(0.6 if (b * 4) % 2 else 0.35), T(b), 0.2 * gain)
            if taiko and b % 4 == 0:
                sc.add("bgm", M.taiko(1.0), T(b), 0.5 * gain)
            b += 0.25

    def arp(b0, b1, gain=0.22):
        r = np.random.default_rng(int(b0 * 10))
        b = b0
        pat = [0, 3, 5, 7, 5, 3, 7, 10]
        i = 0
        while b < b1 - 1e-6:
            m = DMP[pat[i % 8]] + (12 if (i // 8) % 2 else 0)
            sc.add("bgm", M.pluck(m, 0.5, 1.4, 0.25), T(b), gain, pan=-0.3 + 0.6 * (i % 2))
            b += 0.25
            i += 1

    def pad(b0, b1, notes, gain=0.3, kind="pad"):
        d = t(b1) - t(b0) + 0.3
        sc.add("bgm", M.pad(notes, d, 1600, 0.3) if kind == "pad" else M.choir(notes, d, 0.5), T(b0), gain)

    # 国际空间站
    pad(ST[0], ST[1], [50, 57, 62, 65], 0.3)
    arp(ST[0], ST[1], 0.18)
    # 哥伦比亚号：只留低音
    pad(ST[1], ST[2], [38, 45, 50], 0.22)
    # 神舟五号：太鼓 + 箫 + 合唱，重新起势
    sc.add("bgm", M.taiko(1.2), T(ST[2]), 0.7)
    sc.add("bgm", M.gong(5.0), T(ST[2] + 1.5), 0.5)
    roar = M.real_loudest("FX/FX-02/FX-02_01_falcon9_pad_mics_nasa.wav", 4.0)
    x = np.arange(len(roar)) / M.SR
    sc.add("sfx", M.filt(roar, "low", 3500) * np.clip(x / 0.2, 0, 1) * np.clip((4.0 - x) / 1.5, 0, 1), T(ST[2] + 1.5), 0.3)
    pad(ST[2], ST[3], [50, 57, 62, 69], 0.4, "choir")
    for o, m, d in [(1.5, 69, 1.5), (3, 72, 1), (4, 74, 2), (6, 77, 2)]:
        sc.add("bgm", M.xiao(m, t(ST[2] + o + d) - t(ST[2] + o)), T(ST[2] + o), 0.45)
    drums(ST[2] + 4, ST[3], 0.7)
    # 加速快切：鼓 + 琶音 + 合唱，一路推到嫦娥六号
    drums(ST[3], ST[11], 1.0)
    arp(ST[3], ST[12], 0.2)
    prog = [[50, 57, 62, 65], [46, 53, 58, 62], [41, 48, 53, 57], [45, 52, 57, 61]]
    for i, b in enumerate(np.arange(ST[3], ST[12], 4)):
        pad(b, min(b + 4, ST[12]), prog[i % 4], 0.32, "choir")
    for k in range(3, 12):
        sc.add("bgm", M.brass_stab([62, 65, 69, 74], 0.4), T(ST[k]), 0.3)
    # 2021 年春天：竹简声，"天问"落下时的锣
    sc.add("sfx", M.tick() * 3, T(ST[6] + 2.5), 0.3)
    sc.add("bgm", M.gong(4.0, 110), T(ST[6] + 4.5), 0.45)
    # 嫦娥六号：最后一个长音
    sc.add("bgm", M.timpani(38, 2.0, 1.2), T(ST[11]), 0.8)
    sc.add("bgm", M.choir([62, 66, 69, 74], t(ST[12]) - t(ST[11]) + 1.0, 0.3), T(ST[11]), 0.45)
    sc.add("bgm", M.cymbal(3.0), T(ST[11]), 0.2)
    tty = M.real("FX/FX-12/FX-12_01_vintage_teletype_freesound.wav", 0.3, 0.02)
    for k in range(0, 12):
        if k == 1:
            continue
        sc.add("sfx", tty, T(ST[k]), 0.14)


CHAPTER = Chapter("race_b", "竞赛", 128, SHOTS, overlay=overlay, music=music)
