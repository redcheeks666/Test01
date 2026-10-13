"""伍 · 回来，再出发（100 BPM，半速重拍）。现代高清、冷色光效、细线 HUD。

猎鹰 9 号第一次回家 → 发射与着陆两道弧线 → 载人龙飞船 → 筷子夹住助推器（英雄斜坡 + 顿帧）→ 中国火箭回收 →
Artemis II 发射 → 地落 + 阿尔忒弥斯金色线描 → 溅落 → 回到序章：星舰点火、升空、热分离、入轨、部署星链 →
"现在，你知道这一步有多远了"。
"""
import cv2
import numpy as np

from engine.core import (W, H, FPS, CREAM, INK, GOLD, PAPER, PAPER_DARK, SMOKE, chapter_card, post, place, lerp,
                         ease_io, ease_in, ease_out, ease_out_expo, clamp01, glow, caption, grade_clean,
                         grade_teal_orange, draw_text, starfield, flash_at, shake_at, xx, yy, cover_scale, motion_blur,
                         text_width, gray)
from engine.film import Chapter, Shot
from engine.media import Clip, image, retime
from engine.shots import ken, video
from engine import music as M
from chapters import prologue as P
from chapters.race_b import hud, ink_card

HUD_CROP = (0.10, 0.0, 0.90, 0.80)
F9 = Clip("sx0402_f9", "SX/SX-04/SX-04_02_falcon9_first_landing_heli_spacex.mp4", 0.0, 12.6, size=(1920, 1080))
CHOP1 = Clip("sx05_chop1", "SX/SX-05/SX-05_01_starship_flight5_spacex.mp4", 131.5, 141.0, size=(1920, 1080))
CHOP2 = Clip("sx05_chop2", "SX/SX-05/SX-05_01_starship_flight5_spacex.mp4", 145.0, 149.0, size=(1920, 1080))
CZ10 = Clip("cn0705_cz10", "CN/CN-07/CN-07_05_cz10b_sea_net_recovery_cast.mp4", 0.0, 13.5, crop=(0.0, 0.08, 1.0, 0.78),
            size=(720, 1008))
ART_W1 = Clip("na1501_w1", "NA/NA-15/NA-15_01_artemis2_launch_4k_nasa.mp4", 5.5, 9.0, crop=(0.04, 0.06, 0.94, 0.94))
ART_C = Clip("na1501_c", "NA/NA-15/NA-15_01_artemis2_launch_4k_nasa.mp4", 10.5, 16.0, crop=(0.04, 0.06, 0.94, 0.94))
ART_W2 = Clip("na1501_w2b", "NA/NA-15/NA-15_01_artemis2_launch_4k_nasa.mp4", 22.0, 25.0, crop=(0.04, 0.06, 0.94, 0.94))
STAGE = Clip("sx0109_stage", "SX/SX-01/SX-01_09_flight14_hotstaging_angle1_spacex.mp4", 6.5, 14.0)
LIMB = Clip("sx0311_limb", "SX/SX-03/SX-03_11_flight14_earth_limb_cut01_spacex.mp4", 18.0, 34.0, HUD_CROP)
SATS = Clip("sx0310_sats", "SX/SX-03/SX-03_10_flight14_starlink_external_deployment_spacex.mp4", 0.0, 14.0, size=(1920, 1080))
ORBIT = Clip("sx0305_orbit", "SX/SX-03/SX-03_05_flight14_live_orbital_views_spacex.mp4", 2.0, 30.0, HUD_CROP)

COOL = lambda x: grade_teal_orange(x, 0.25)     # noqa: E731


def s_card(c):
    return chapter_card(c, "回来，再出发")


# 猎鹰 9 号：触地落在第 6 拍（3.6 秒），着陆反推放慢到 0.35 倍
_F9K = [(0, 1.2), (2.8, 1.2), (3.4, 0.35), (4.2, 0.35), (4.8, 0.8)]
F9_T = retime(_F9K, start=9.0 - retime(_F9K)(3.6))


def s_falcon(c):
    f = video(c, F9, max(F9_T(c.lsec), 0), grade=COOL, zoom=lerp(1.05, 1.15, c.k))
    caption(f, c, 0.5, [("回家", "big"), ("火箭第一次垂直落回着陆场", "mid"), ("2015.12.21 · 猎鹰 9 号", "small")])
    sh, e = shake_at(c, 6.0, 12, 0.6)
    return post(f, c.fi, shake=sh, grain=0.02, flash=flash_at(c, 6.0, 1, 0.35))


def s_arcs(c):
    src = grade_clean(image("SX-04", "01", 3000), 0.2, 1.08, 1.1)
    f = ken(c, src, (0.55, 0.55, 1.12), (0.5, 0.5, 1.02))
    # 光迹描绘：沿弧线方向从左下往右上逐渐显现
    k = ease_io(clamp01(c.lb / 5.0))
    edge = lerp(-0.2, 1.2, k)
    u = (xx / W * 0.6 + (1 - yy / H) * 0.4)
    m = np.clip((edge - u) / 0.08, 0, 1)[..., None]
    dark = f * 0.18
    f = dark * (1 - m) + f * m + glow(0, 0, 1, (0, 0, 0)) * 0
    f = hud(f, 1.0, "ORBCOMM-2 · LZ-1")
    caption(f, c, 0.6, [("两道弧线", "big"), ("一道飞上去，一道落回来", "mid")])
    return post(f, c.fi, grain=0.02)


def s_demo2(c):
    src = grade_clean(image("SX-07", "03", 3000), 0.15, 1.06, 1.05)
    f = ken(c, src, (0.62, 0.62, 1.15), (0.66, 0.30, 1.6))
    caption(f, c, 0.4, [("载人龙飞船", "big"), ("商业飞船第一次把航天员送上太空", "mid"), ("2020.05.30", "small")])
    return post(f, c.fi, grain=0.02)


# 筷子：合拢的特写落在第 8 拍（4.8 秒），0.4 倍慢放，夹住时顿帧
_CHK = [(0, 1.3), (3.6, 1.3), (4.2, 0.4), (5.6, 0.4), (6.0, 1.0)]
CH_T = retime(_CHK, start=(139.3 - 131.5) - retime(_CHK)(4.8))


def s_chop(c):
    t_ = c.lsec
    if t_ < 6.0:
        tt = t_ if t_ < 4.8 else (4.8 if t_ < 4.9 else t_ - 0.1)     # 顿帧 3 帧
        f = video(c, CHOP1, max(CH_T(tt), 0), grade=COOL, zoom=1.05)
    else:
        f = video(c, CHOP2, (t_ - 6.0) * 1.0, grade=COOL, zoom=lerp(1.0, 1.06, (t_ - 6.0) / 2.4))
    caption(f, c, 8.3, [("筷子", "big"), ("发射塔用机械臂接住了火箭", "mid"), ("2024.10.13 · 星舰第 5 次试飞", "small")])
    sh, e = shake_at(c, 8.0, 18, 0.7)
    return post(f, c.fi, shake=sh, chroma=5 * e, flash=flash_at(c, 8.0, 2, 0.5), grain=0.02)


def s_china(c):
    L = c.lb
    if L < 3:
        f = ink_card(c, image("CN-07", "04", 1600), 1240, 560, 1100, clamp01(L / 0.8))
        cap = [("中国火箭回收", "big"), ("2024 年起，一次次垂直起降试验", "mid")]
    elif L < 6:
        f = ink_card(c, image("CN-07", "03", 1600), 1240, 560, 1100, clamp01((L - 3) / 0.6))
        cap = [("中国火箭回收", "big"), ("2024 年起，一次次垂直起降试验", "mid")]
    elif L < 12:
        f = PAPER.copy()
        v = CZ10.at(c.since(6) * 1.15)
        h, w = v.shape[:2]
        s = 860 / h
        rgb, a = place(v, w / 2, h / 2, s, 0, 1300, 540)
        rev = np.clip((SMOKE[1] - lerp(1.1, -0.1, ease_io(clamp01((L - 6) / 0.8)))) * 5, 0, 1)[..., None]
        f = f * (1 - a * rev) + rgb * a * rev
        cap = [("可控回收", "big"), ("2026.07 · 长征十号乙一级海上网系回收", "mid"), ("我国首次成功实施运载火箭一级可控回收", "small")]
    else:
        f = ink_card(c, image("CN-07", "01", 1600), 1240, 560, 1100, clamp01((L - 12) / 0.6))
        cap = [("落地", "big"), ("2026.08 · 朱雀三号一级陆地回收", "mid")]
    at = {0: 0.3, 3: 3.3, 6: 6.3, 12: 12.3}[0 if L < 3 else 3 if L < 6 else 6 if L < 12 else 12]
    if L >= 3 and L < 6:
        at = 0.3
    caption(f, c, at, cap, color=INK, scrim=0.0)
    return post(f, c.fi, grain=0.02)


def s_artemis_launch(c):
    t_ = c.lsec
    if t_ < 1.2:
        f = video(c, ART_W1, 0.5 + t_, grade=COOL, zoom=1.05)
    elif t_ < 3.6:
        u = t_ - 1.2
        st = 0.5 + (u * 0.45 if u < 1.6 else 0.72 + (u - 1.6) * 1.0)
        f = video(c, ART_C, st, grade=COOL, zoom=lerp(1.0, 1.08, u / 2.4))
    else:
        f = video(c, ART_W2, 0.05 + (t_ - 3.6) * 1.0, grade=COOL, zoom=lerp(1.0, 1.15, (t_ - 3.6) / 2.4),
                  cy=lerp(0.55, 0.42, (t_ - 3.6) / 2.4))
    caption(f, c, 2.4, [("Artemis II", "big"), ("2026 年 4 月 · 肯尼迪航天中心", "mid")])
    sh, e = shake_at(c, 2.0, 22, 1.0)
    return post(f, c.fi, shake=sh, chroma=5 * e, flash=flash_at(c, 2.0, 2, 0.8), grain=0.02)


_GOLD = None


def _gold_artemis():
    """把陶瓶上的阿尔忒弥斯画像描成金色线描（边缘检测）。"""
    global _GOLD
    if _GOLD is None:
        v = image("PD-13", "01", 2400)
        h, w = v.shape[:2]
        crop = v[int(0.38 * h):int(0.80 * h), int(0.25 * w):int(0.78 * w)]
        g = (gray(crop) * 255).astype(np.uint8)
        e = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 2.0), 40, 110)
        e = cv2.dilate(e, np.ones((2, 2), np.uint8))
        _GOLD = cv2.GaussianBlur(e.astype(np.float32) / 255, (0, 0), 0.8)
    return _GOLD


def s_earthset(c):
    L = c.lb
    if L < 4:
        f = ken(c, grade_clean(image("NA-02", "03", 3000), 0.1, 1.06, 1.05), (0.5, 0.45, 1.25), (0.5, 0.4, 1.12), t0=0, t1=4)
    else:
        f = ken(c, grade_clean(image("NA-02", "02", 3000), 0.1, 1.06, 1.05), (0.5, 0.55, 1.08), (0.5, 0.58, 1.2), t0=4, t1=12)
        mix = clamp01((L - 4) / 0.8)
        if mix < 1:
            prev = ken(c, grade_clean(image("NA-02", "03", 3000), 0.1, 1.06, 1.05), (0.5, 0.4, 1.12), (0.5, 0.4, 1.12))
            f = prev * (1 - mix) + f * mix
    if L >= 7.0:
        e = _gold_artemis()
        s = 620 / e.shape[0]
        er = cv2.resize(e, (int(e.shape[1] * s), 620))
        k = ease_io(clamp01((L - 7.0) / 2.5))
        er = er.copy()
        er[int(er.shape[0] * k):] = 0
        x0, y0 = 1180, 120
        h_, w_ = er.shape
        reg = f[y0:y0 + h_, x0:x0 + w_]
        a = er[..., None] * 0.9
        f[y0:y0 + h_, x0:x0 + w_] = reg * (1 - a) + np.array(GOLD) * a + glow(0, 0, 1, (0, 0, 0))[y0:y0 + h_, x0:x0 + w_]
    caption(f, c, 0.5, [("四名航天员绕月", "big"), ("飞得比任何人类都远", "mid"), ("半个多世纪后，人类重返月球附近", "small")],
            out=6.6)
    caption(f, c, 7.6, [("阿尔忒弥斯", "big"), ("是阿波罗的孪生姐姐", "mid")])
    return post(f, c.fi, grain=0.02)


def s_splash(c):
    if c.lb < 5:
        f = ken(c, grade_clean(image("NA-17", "01", 3000), 0.15, 1.05, 1.05), (0.5, 0.22, 2.0), (0.5, 0.45, 1.5), t0=0, t1=5)
    else:
        f = ken(c, grade_clean(image("NA-17", "02", 3000), 0.15, 1.05, 1.05), (0.5, 0.6, 1.2), (0.5, 0.62, 1.1), t0=5, t1=8)
    caption(f, c, 0.4, [("回家", "big"), ("太平洋溅落", "mid")])
    sh, e = shake_at(c, 5.0, 10, 0.5)
    return post(f, c.fi, shake=sh, flash=flash_at(c, 5.0, 1, 0.3), grain=0.02)


# ---------------- 回到序章 ----------------

def s_pad_again(c):
    f = P.s_pad_photo(c)
    n = int(clamp01(c.since(0.3) / 1.4) * len(P.DATE) + 1e-6)
    draw_text(f, P.DATE[:n], "mono", 54, CREAM, W / 2 - text_width(P.DATE, "mono", 54, 6) / 2, 90, tracking=6)
    return f


_IG2K = [(0, 1.0), (1.32, 1.0), (1.5, 0.3), (2.64, 0.3), (3.18, 1.5), (3.6, 1.5)]
IGN2 = retime(_IG2K, start=(15.0 - 10.0) - retime(_IG2K)(1.5))


def s_ignite_again(c):
    """与序章同一机位、同一变速，只是按本章速度缩放：强拍白闪对准火焰爆开。"""
    st = IGN2(c.lsec)
    f = video(c, P.PAD, st, grade=lambda x: grade_teal_orange(x, 0.3), zoom=1.0 + 0.04 * ease_io(c.k), cy=0.48)
    if 2.0 <= c.lb < 2.5:
        f *= lerp(1.0, 0.75, (c.lb - 2.0) / 0.5)
    sh, e = shake_at(c, 2.5, 26, 1.4)
    return post(f, c.fi, flash=flash_at(c, 2.5, 2, 0.9), shake=sh, chroma=8 * e, grain=0.02)


def s_ascent_again(c):
    st = P.ASC(c.lsec)
    z = 1.0 + 0.3 * ease_in(clamp01((c.lb - 2.0) / 2.0), 2)
    f = video(c, P.AIR, st, grade=COOL, zoom=z, cy=lerp(0.5, 0.3, ease_in(c.k, 2)))
    if c.lb > 2.4:
        f = motion_blur(f, 70 * ease_in((c.lb - 2.4) / 1.6, 2), 90)
    f = f * (1 - 0) + clamp01((c.lb - 3.3) / 0.7)
    return post(np.clip(f, 0, 1), c.fi, grain=0.02)


def s_staging(c):
    f = video(c, STAGE, 1.5 + c.lsec * 1.2, grade=COOL, zoom=1.04)
    w = 1 - clamp01(c.lsec / 0.4)
    f = f * (1 - w) + w
    caption(f, c, 0.6, [("热分离", "big"), ("二级在一级还在推进时点火", "mid")])
    sh, e = shake_at(c, 1.0, 14, 0.6)
    return post(f, c.fi, shake=sh, chroma=4 * e, grain=0.02)


def s_orbit(c):
    if c.lb < 8:
        f = video(c, LIMB, c.lsec * 0.7, grade=COOL, zoom=lerp(1.0, 1.06, c.lb / 8))
    else:
        f = video(c, SATS, 1.0 + c.since(8) * 0.8, grade=COOL, zoom=lerp(1.0, 1.06, (c.lb - 8) / 8))
        f = f * clamp01(c.since(8) / 0.3 + 0.0) if c.since(8) < 0 else f
    caption(f, c, 0.6, [("第一次，进入轨道", "big"), ("2026.09.28 · 星舰第 14 次试飞", "mid")], out=7.4)
    caption(f, c, 8.6, [("部署星链卫星", "big"), ("26 颗", "mid")])
    return post(f, c.fi, grain=0.02)


def s_far(c):
    f = video(c, ORBIT, c.lsec * 0.6, grade=COOL, zoom=lerp(1.0, 1.08, c.k))
    a = clamp01((c.lb - 1.0) / 1.2)
    f *= 1 - 0.45 * a
    draw_text(f, "现在，你知道这一步有多远了", "serif", 58, CREAM, W / 2, H / 2, a, anchor="c", tracking=10)
    f = f * (1 - clamp01((c.lb - (c.beats - 1.5)) / 1.5))
    return post(f, c.fi, grain=0.02)


SHOTS = [Shot(8, s_card, name="5-01"), Shot(8, s_falcon, name="5-02"), Shot(8, s_arcs, name="5-02b"),
         Shot(6, s_demo2, name="5-03"), Shot(14, s_chop, name="5-04"), Shot(16, s_china, name="5-05"),
         Shot(10, s_artemis_launch, name="5-06a"), Shot(12, s_earthset, name="5-06b"), Shot(8, s_splash, name="5-06c"),
         Shot(5, s_pad_again, name="5-07a"), Shot(6, s_ignite_again, name="5-07b"), Shot(4, s_ascent_again, name="5-07c"),
         Shot(4, s_staging, name="5-07d"), Shot(16, s_orbit, name="5-07e"), Shot(10, s_far, name="5-07f")]
ST = np.concatenate([[0], np.cumsum([s.beats for s in SHOTS])])


def music(sc, ch):
    T, t = ch.T, ch.t

    def drums(b0, b1, gain=1.0):
        """半速重拍：底鼓在 1、3 拍，军鼓 + 太鼓砸在 2、4 拍。"""
        b = b0
        while b < b1 - 1e-6:
            if b % 2 == 0:
                sc.add("bgm", M.kick(1.0), T(b), 0.6 * gain)
            if b % 2 == 1:
                sc.add("bgm", M.snare(1.0), T(b), 0.35 * gain)
                sc.add("bgm", M.taiko(0.9, 70), T(b), 0.45 * gain)
            sc.add("bgm", M.hat(0.5), T(b + 0.5), 0.15 * gain)
            b += 1.0

    def ost16(b0, b1, roots, gain=0.26):
        b = b0
        while b < b1 - 1e-6:
            r = roots[int((b - b0) // 4) % len(roots)]
            pat = [0, 12, 7, 12, 0, 12, 10, 12]
            m = r + pat[int(round((b % 2) * 4)) % 8]
            sc.add("bgm", M.strings_short(m, (t(b + 0.25) - t(b)) * 0.8), T(b), gain, pan=-0.2)
            b += 0.25

    def chord(b0, b1, notes, gain=0.32, kind="pad"):
        d = t(b1) - t(b0) + 0.4
        sc.add("bgm", M.pad(notes, d, 1800, 0.3) if kind == "pad" else M.choir(notes, d, 0.5), T(b0), gain)

    def hit(b, gain=1.0, low=True):
        sc.add("bgm", M.timpani(38, 2.0, 1.2), T(b), 0.8 * gain)
        sc.add("bgm", M.brass_stab([62, 65, 69, 74, 50], 1.0), T(b), 0.55 * gain)
        sc.add("bgm", M.cymbal(2.5), T(b), 0.2 * gain)
        if low:
            sc.add("sfx", M.real("FX/FX-11/FX-11_02_cinematic_low_hit_pixabay.mp3"), T(b), 0.45 * gain)

    def roar(b, dur, gain=0.5):
        r = M.real_loudest("FX/FX-02/FX-02_01_falcon9_pad_mics_nasa.wav", dur)
        x = np.arange(len(r)) / M.SR
        sc.add("sfx", M.filt(r, "low", 4500) * np.clip(x / 0.06, 0, 1) * np.clip((dur - x) / 1.5, 0, 1), T(b), gain)

    sc.add("sfx", M.stamp(), T(2.5), 0.7)
    chord(0, 8, [38, 45, 50, 57], 0.25)
    # 猎鹰 9 号 → 弧线 → 龙飞船
    drums(ST[1], ST[4])
    ost16(ST[1], ST[4], [38, 34, 41, 36])
    chord(ST[1], ST[2], [50, 57, 62, 65], 0.3)
    roar(ST[1] + 3.5, 3.0, 0.35)
    hit(ST[1] + 6)
    chord(ST[2], ST[3], [62, 65, 69, 74], 0.4, "choir")
    for i in range(16):
        sc.add("bgm", M.bell([74, 77, 81, 86][i % 4], 1.4), T(ST[2] + i * 0.5), 0.13)
    roar(ST[3], 3.0, 0.3)
    chord(ST[3], ST[4], [58, 62, 65, 70], 0.35, "choir")
    # 筷子：军鼓滚奏 + 上升音推到第 8 拍
    chord(ST[4], ST[5], [45, 52, 57, 61], 0.3)
    ost16(ST[4], ST[4] + 7.5, [33, 33], 0.3)
    sc.add("sfx", M.riser(t(ST[4] + 7.9) - t(ST[4] + 4)), T(ST[4] + 4), 0.35)
    b = ST[4] + 5
    while b < ST[4] + 7.9:
        sc.add("bgm", M.snare(0.3 + 0.7 * (b - ST[4] - 5) / 3), T(b), 0.3)
        b += 0.25 if b < ST[4] + 7 else 0.125
    sc.gate(T(ST[4] + 7.75), T(ST[4] + 8), 0.1)
    hit(ST[4] + 8, 1.15)
    sc.add("sfx", M.boom(2.5, 50, 26, 0.15), T(ST[4] + 8), 0.4)
    drums(ST[4] + 8, ST[5])
    chord(ST[4] + 8, ST[5], [62, 65, 69, 74], 0.45, "choir")
    # 中国火箭回收：太鼓 + 箫 + 弦乐
    drums(ST[5], ST[6], 0.9)
    ost16(ST[5], ST[6], [38, 34, 36, 33])
    chord(ST[5], ST[6], [50, 57, 62, 69], 0.32, "choir")
    for o, m, d in [(0.5, 69, 1.5), (2, 72, 1), (3, 74, 3), (6, 77, 2), (8, 74, 2), (10, 72, 2), (12, 74, 4)]:
        sc.add("bgm", M.xiao(m, t(ST[5] + o + d) - t(ST[5] + o)), T(ST[5] + o), 0.4)
    sc.add("bgm", M.gong(4.0), T(ST[5] + 6), 0.4)
    # Artemis II：点火 → 地落（全编制旋律）→ 金色阿尔忒弥斯 → 溅落
    sc.gate(T(ST[6] + 1.75), T(ST[6] + 2), 0.1)
    hit(ST[6] + 2, 1.1)
    roar(ST[6] + 2, 5.0, 0.5)
    drums(ST[6] + 2, ST[7])
    ost16(ST[6] + 2, ST[8], [38, 34, 41, 36], 0.3)
    chord(ST[6] + 2, ST[7], [62, 65, 69, 74], 0.45, "choir")
    chord(ST[7], ST[8], [50, 57, 62, 65, 69], 0.4, "choir")
    drums(ST[7], ST[8], 0.8)
    mel = [(0, 69, 2), (2, 74, 2), (4, 72, 1), (5, 74, 1), (6, 77, 4), (10, 76, 2)]
    for o, m, d in mel:
        sc.add("bgm", M.horn(m, t(ST[7] + o + d) - t(ST[7] + o)), T(ST[7] + o), 0.38)
    sc.add("bgm", M.bell(81, 3.5), T(ST[7] + 7), 0.3)
    sc.add("bgm", M.gong(5.0, 100), T(ST[7] + 7), 0.35)
    chord(ST[8], ST[9], [46, 53, 58, 62], 0.3)
    hit(ST[8] + 5, 0.7)
    # 回到序章：同样的打字声与心跳，这次不倒带
    key = M.real("FX/FX-01/FX-01_01_mech_keyboard_click_pixabay.mp3", fade=0.003)
    for i in range(len(P.DATE)):
        sc.add("sfx", key, T(ST[9] + 0.3) + i * (t(1.4) / len(P.DATE)), 0.4)
    for b in np.arange(ST[9], ST[10], 1.0):
        sc.add("bgm", M.heartbeat(), T(b), 0.55)
    chord(ST[9], ST[10] + 2, [38, 45, 50], 0.3)
    sc.gate(T(ST[10] + 2.0), T(ST[10] + 2.5), 0.06)
    hit(ST[10] + 2.5, 1.2)
    roar(ST[10] + 2.5, 8.0, 0.6)
    drums(ST[10] + 2.5, ST[14])
    ost16(ST[10] + 2.5, ST[14], [38, 34, 41, 36], 0.3)
    chord(ST[10] + 2.5, ST[12], [50, 57, 62, 65, 69], 0.45, "choir")
    sc.add("sfx", M.riser(t(ST[12]) - t(ST[11])), T(ST[11]), 0.35)
    hit(ST[12], 1.0)
    sc.add("sfx", M.boom(2.0, 60, 30, 0.15), T(ST[12] + 1), 0.35)
    # 入轨：全片最高潮
    hit(ST[13], 1.2)
    chord(ST[13], ST[14], [62, 65, 69, 74, 77], 0.5, "choir")
    chord(ST[13], ST[15], [38, 45, 50, 57], 0.4)
    mel2 = [(0, 74, 2), (2, 77, 2), (4, 81, 4), (8, 79, 2), (10, 77, 2), (12, 74, 4)]
    for o, m, d in mel2:
        sc.add("bgm", M.horn(m, t(ST[13] + o + d) - t(ST[13] + o)), T(ST[13] + o), 0.4)
        sc.add("bgm", M.horn(m - 12, t(ST[13] + o + d) - t(ST[13] + o)), T(ST[13] + o), 0.25)
    # 最后一句：长音收住
    chord(ST[14], ST[15], [50, 57, 62, 66, 69], 0.4, "choir")
    sc.add("bgm", M.timpani(38, 3.0, 0.9), T(ST[14]), 0.6)


CHAPTER = Chapter("return_", "回来，再出发", 100, SHOTS,
                  clips=[F9, CHOP1, CHOP2, CZ10, ART_W1, ART_C, ART_W2, STAGE, LIMB, SATS, ORBIT, P.PAD, P.AIR],
                  music=music)
