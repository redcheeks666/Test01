"""序章 · 冷开场（60 BPM，一拍一秒）。

黑场星点 → 打字出日期 → 塔旁日出 → 点火（吸气半拍、强拍白闪、0.3 倍慢放）→ 斜坡加速升空 →
VHS 倒带、年份倒数 → 星空"这一步，人类走了五千年" → 真实满月升起，片名"天问"。
"""
import cv2
import numpy as np

from engine.core import (W, H, FPS, CREAM, INK, GOLD, PAPER, starfield, draw_text, post, lerp, ease_io, ease_out,
                         ease_in, ease_out_expo, clamp01, flash_at, shake_at, grade_teal_orange, motion_blur, cam,
                         place, glow, text_width, SEAL, fbm, GRAIN, caption)
from engine.film import Chapter, Shot
from engine.media import Clip, image, retime
from engine.shots import ken, video
from engine import music as M

HUD_CROP = (0.10, 0.0, 0.90, 0.80)       # 裁掉 SpaceX 直播画面下缘的遥测条
PAD = Clip("sx0102_pad_v2", "SX/SX-01/SX-01_02_flight14_liftoff_spacex.mp4", 10.0, 17.3, HUD_CROP)
AIR = Clip("sx0102_air_v2", "SX/SX-01/SX-01_02_flight14_liftoff_spacex.mp4", 22.9, 34.0, HUD_CROP)
STARS = starfield(5)
DATE, PLACE = "2026.09.28", "美国得克萨斯 · 星际基地"

# 点火镜头的变速：第 2.5 秒（强拍）对准源视频第 14.6 秒火焰爆开
_IGN_KEYS = [(0, 1.0), (2.2, 1.0), (2.5, 0.3), (4.4, 0.3), (5.3, 1.5), (6.0, 1.5)]
_f0 = retime(_IGN_KEYS)
IGN = retime(_IGN_KEYS, start=(15.0 - 10.0) - _f0(2.5))
ASC = retime([(0, 1.5), (3.0, 1.5), (4.0, 2.6)], start=0.3)


def _stars_in(c):
    return post(STARS * clamp01(c.lsec / 1.2), c.fi, grain=0.02, vignette=False, weave=False)


def s_black(c):
    return _stars_in(c)


def s_date(c):
    frame = STARS.copy()
    n1 = int(clamp01(c.since(0.3) / 1.6) * len(DATE) + 1e-6)
    n2 = int(clamp01(c.since(2.0) / 1.4) * len(PLACE) + 1e-6)
    draw_text(frame, DATE[:n1], "mono", 78, CREAM, W / 2 - text_width(DATE, "mono", 78, 8) / 2, H / 2 - 70, tracking=8)
    if n2:
        draw_text(frame, PLACE[:n2], "serif", 34, (0.8, 0.78, 0.72), W / 2, H / 2 + 60, anchor="c", tracking=6)
    # 光标闪烁
    if c.lb < 3.6 and int(c.lsec * 2) % 2 == 0:
        x = W / 2 - text_width(DATE, "mono", 78, 8) / 2 + text_width(DATE[:n1] or " ", "mono", 78, 8) + 12
        frame[int(H / 2 - 60):int(H / 2 + 30), int(x):int(x + 6)] = CREAM
    return post(frame, c.fi, grain=0.02, vignette=False, weave=False)


def s_pad_photo(c):
    src = image("SX-02", "01")
    f = ken(c, src, (0.50, 0.55, 1.12), (0.50, 0.52, 1.20))
    f = grade_teal_orange(f, 0.3) * clamp01(c.lsec / 0.5)
    return post(f, c.fi, grain=0.025)


def s_ignite(c):
    st = IGN(c.lsec)
    z = 1.0 + 0.04 * ease_io(c.k)
    f = video(c, PAD, st, grade=lambda x: grade_teal_orange(x, 0.3), zoom=z, cy=0.48)
    if 2.0 <= c.lb < 2.5:            # 吸气：压暗
        f *= lerp(1.0, 0.75, (c.lb - 2.0) / 0.5)
    sh, e = shake_at(c, 2.5, 26, 1.4)
    caption(f, c, 2.8, [("124 米", "big"), ("人类造过最大的火箭", "mid")])
    return post(f, c.fi, flash=flash_at(c, 2.5, 2, 0.9), shake=sh, chroma=8 * e, grain=0.025)


def s_ascent(c):
    st = ASC(c.lsec)
    z = 1.0 + 0.25 * ease_in(clamp01((c.lb - 2.6) / 1.4), 2)
    f = video(c, AIR, st, grade=lambda x: grade_teal_orange(x, 0.3), zoom=z, cy=lerp(0.5, 0.35, ease_in(c.k, 2)))
    if c.lb > 2.8:
        f = motion_blur(f, 60 * ease_in((c.lb - 2.8) / 1.2, 2), 90)
    caption(f, c, 0.2, [("第一次", "big"), ("进入地球轨道", "mid")])
    sh, e = shake_at(c, 0, 10, 0.5)
    return post(f, c.fi, shake=sh, grain=0.025)


def _vhs(frame, fi, amt):
    """VHS 倒带：扫描线、横向撕裂、色散、噪声带。"""
    r = np.random.default_rng(fi)
    out = frame.copy()
    for _ in range(int(3 + 10 * amt)):
        y0 = int(r.uniform(0, H - 30))
        hgt = int(r.uniform(4, 40))
        dx = int(r.normal(0, 40 * amt))
        out[y0:y0 + hgt] = np.roll(out[y0:y0 + hgt], dx, axis=1)
    c = int(4 + 10 * amt)
    out[..., 0] = np.roll(out[..., 0], c, 1)
    out[..., 2] = np.roll(out[..., 2], -c, 1)
    out[::3] *= 0.82
    band = int(r.uniform(0, H))
    out[band:band + 26] = out[band:band + 26] * 0.5 + r.random((min(26, H - band), W, 1)) * 0.5
    return out


def s_rewind(c):
    t = c.lsec
    if t < 1.2:                                  # 升空画面倒放
        st = ASC(4.0) - (ASC(4.0) - 0.3) * ease_in(t / 1.2, 1.6)
        f = video(c, AIR, st, grade=lambda x: grade_teal_orange(x, 0.3), blend=False)
    elif t < 2.4:                                # 点火画面倒放
        st = IGN(6.0) - (IGN(6.0) - 0.0) * ease_in((t - 1.2) / 1.2, 1.6)
        f = video(c, PAD, st, grade=lambda x: grade_teal_orange(x, 0.3), blend=False)
    else:
        f = STARS.copy() * 0.6
    # 卡顿：按十六分音符重复上一帧
    f = _vhs(f, c.fi // 2 if int(c.lb * 4) % 2 else c.fi, 0.4 + 0.6 * c.k)
    # 年份倒数
    yr = int(lerp(2026, -3000, ease_in(clamp01(t / 2.7), 2.2)))
    label = f"{yr}" if yr > 0 else "公元前"
    big = 150
    draw_text(f, label, "mono" if yr > 0 else "brush", big, CREAM, W / 2, H / 2, 1.0, anchor="c", tracking=6)
    if t > 2.75:
        f *= 0
    return post(f, c.fi, grain=0.05, vignette=False, weave=False)


def s_five_thousand(c):
    f = STARS.copy()
    a = clamp01(c.since(0.6) / 0.8) * (1 - clamp01((c.lb - 2.6) * 3))
    draw_text(f, "这一步，人类走了五千年", "serif", 52, CREAM, W / 2, H / 2, a, anchor="c", tracking=10)
    return post(f, c.fi, grain=0.02, vignette=False, weave=False)


_MOON = None


def _moon():
    global _MOON
    if _MOON is None:
        m = image("NA-01", "01", 2400)                     # 2400×1350，月面半径约 0.47 高
        h, w = m.shape[:2]
        r = 0.47 * h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = np.clip((r - np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)) / 2 + 0.5, 0, 1)
        _MOON = (m, a, r)
    return _MOON


def moon_layer(cy_screen, radius_screen, base):
    m, a, r = _moon()
    s = radius_screen / r
    rgb, al = place(m * np.array([1.0, 0.97, 0.9], np.float32), m.shape[1] / 2, m.shape[0] / 2, s, 0, W / 2, cy_screen)
    al2, _ = place(a[..., None].repeat(3, 2), m.shape[1] / 2, m.shape[0] / 2, s, 0, W / 2, cy_screen)
    al = al2[..., :1]
    halo = glow(W / 2, cy_screen, radius_screen * 1.25, (0.35, 0.32, 0.26), 0.35)
    return base * (1 - al) + rgb * al + halo * (1 - al)


def s_title(c):
    rise = ease_out(clamp01(c.lb / 3.0), 3)
    cy = lerp(H + 380, H / 2 - 20, rise)
    f = moon_layer(cy, 330, STARS.copy())
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
        draw_text(f, "人类向往星空的五千年", "serif", 36, CREAM, W / 2, H / 2 + 380, clamp01(c.since(4.0)), anchor="c",
                  tracking=12)
    out = 1 - clamp01((c.lb - (c.beats - 0.8)) / 0.8)
    return post(f * out, c.fi, grain=0.02, vignette=False, weave=False)


SHOTS = [Shot(1.5, s_black, name="P-01"), Shot(4, s_date, name="P-02"), Shot(2, s_pad_photo, name="P-03"),
         Shot(6, s_ignite, name="P-04"), Shot(4, s_ascent, name="P-05"), Shot(3, s_rewind, name="P-06"),
         Shot(3, s_five_thousand, name="P-07"), Shot(7, s_title, name="P-08")]


def music(sc, ch):
    T = ch.T
    sc.add("bgm", M.drone(33, ch.t(10) + 0.5), T(0), 0.35)
    # 打字：每个字符一下，回车收尾
    key = M.real("FX/FX-01/FX-01_01_mech_keyboard_click_pixabay.mp3", fade=0.003)
    for i in range(len(DATE)):
        sc.add("sfx", key, T(1.5 + 0.3) + i * 1.6 / len(DATE), 0.5, pan=M.rng.uniform(-0.3, 0.3))
    for i in range(0, len(PLACE), 2):
        sc.add("sfx", key, T(1.5 + 2.0) + i * 1.4 / len(PLACE), 0.35, pan=M.rng.uniform(-0.3, 0.3))
    sc.add("sfx", M.real("FX/FX-01/FX-01_03_mechanical_enter_key_freesound.wav", fade=0.003), T(5.1), 0.6)
    # 心跳
    for b in np.arange(5.5, 9.5, 1.0):
        sc.add("bgm", M.heartbeat(), T(b), 0.5 + 0.1 * (b - 5.5))
    sc.add("bgm", M.riser(ch.t(9.5) - ch.t(7.5)), T(7.5), 0.25)
    sc.gate(T(9.5), T(10.0), 0.06)
    # 点火：真实轰鸣 + 低频冲击（压低高频爆裂，防刺耳）+ 定音鼓 + 铜管 + 宏大和弦
    t_ign = T(10.0)
    roar = M.real_loudest("FX/FX-02/FX-02_01_falcon9_pad_mics_nasa.wav", 7.5)
    x = np.arange(len(roar)) / M.SR
    roar = M.filt(roar, "low", 5000) * np.clip(x / 0.06, 0, 1) * np.clip((7.5 - x) / 2.0, 0, 1)
    sc.add("sfx", roar, t_ign, 0.7)
    sc.add("sfx", M.boom(3.0, 50, 26, 0.25), t_ign, 0.6)
    sc.add("bgm", M.timpani(38, 2.0, 1.2), t_ign, 0.8)
    sc.add("bgm", M.brass_stab([62, 65, 69, 74, 50], 1.0), t_ign, 0.6)
    sc.add("bgm", M.cymbal(3.0), t_ign, 0.2)
    for b0, b1, ch_ in [(10, 13.5, [50, 57, 62, 65, 69]), (13.5, 17.5, [46, 53, 58, 62, 65])]:
        d = ch.t(b1) - ch.t(b0)
        sc.add("bgm", M.pad(ch_, d + 0.3, 1800, 0.3), T(b0), 0.6)
        sc.add("bgm", M.choir([m + 12 for m in ch_[1:4]], d + 0.3, 0.5), T(b0), 0.35)
        for m in ch_[:2]:
            sc.add("bgm", M.strings_long(m - 12, d), T(b0), 0.35)
    sc.add("bgm", M.timpani(34, 2.0, 1.1), T(13.5), 0.8)
    sc.add("bgm", M.brass_stab([58, 62, 65, 70, 46], 0.8), T(13.5), 0.55)
    for k, m in enumerate([69, 72, 74, 77]):
        sc.add("bgm", M.horn(m, 0.9), T(14 + k * 0.8), 0.3)
    # 倒带：磁带倒带 + VHS 噪声 + 倒放的和弦，第 20.5 拍前一刻切断
    sc.add("sfx", M.real("FX/FX-03/FX-03_01_reel_tape_rewind_pixabay.mp3", ch.t(20.4) - ch.t(17.5), 0.02), T(17.5), 0.75)
    sc.add("sfx", M.real("FX/FX-17/FX-17_02_vhs_damaged_tape_freesound.wav", ch.t(20.4) - ch.t(17.5), 0.02), T(17.5), 0.35)
    rev = M.pad([46, 53, 58, 62, 65], 3.0, 2200, 0.05)[::-1] * np.linspace(0.3, 1, int(3.0 * M.SR))
    sc.add("bgm", rev, T(17.5), 0.5)
    sc.silence(T(20.4), T(20.55))
    # 只剩一个高音
    sc.add("bgm", M.strings_long(81, ch.t(23.5) - ch.t(20.55), 3000), T(20.55), 0.25)
    sc.add("bgm", M.xiao(69, 2.6), T(20.9), 0.2)
    # 月亮升起，片名：锣 + 箫
    sc.add("bgm", M.pad([38, 45, 50, 57], ch.t(30.5) - ch.t(23.5), 1000, 1.5), T(23.5), 0.4)
    sc.add("bgm", M.gong(7.0), T(26.5), 0.9)
    for b0, m, d in [(27.0, 62, 1.5), (28.5, 65, 0.8), (29.3, 67, 0.8), (30.1, 69, 2.0)]:
        sc.add("bgm", M.xiao(m, d), T(b0), 0.5)


CHAPTER = Chapter("prologue", "序章", 60, SHOTS, clips=[PAD, AIR], music=music)
