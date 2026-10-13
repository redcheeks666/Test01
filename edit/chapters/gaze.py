"""贰 · 仰望（84 BPM）。张衡浑天仪（代码绘制）→ 伽利略（木刻 → 真实上弦月，木星卫星）→
开普勒（扉页 → 椭圆轨道）→ 牛顿（原书插图 → 炮弹绕地 → 国际空间站视角的真实地球）。"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from engine.core import (W, H, FPS, CREAM, INK, GOLD, PAPER, PAPER_DARK, chapter_card, post, place, lerp, ease_io,
                         ease_out, ease_in, clamp01, fbm, glow, caption, circle_mask, xx, yy, duotone, starfield,
                         draw_text, cover_scale, fit_scale)
from engine.film import Chapter, Shot
from engine.media import Clip, image
from engine.shots import ken, video
from engine import music as M

ISS = Clip("na0402_iss", "NA/NA-04/NA-04_02_iss_view_of_earth_nasa.mp4", 20.0, 34.0)
BRONZE = (196, 150, 86)
S = 2


def s_card(c):
    return chapter_card(c, "仰望")


# ---------------- 张衡浑天仪 ----------------

def _rot(ax, a):
    c_, s_ = np.cos(a), np.sin(a)
    if ax == "x":
        return np.array([[1, 0, 0], [0, c_, -s_], [0, s_, c_]])
    if ax == "y":
        return np.array([[c_, 0, s_], [0, 1, 0], [-s_, 0, c_]])
    return np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])


def _sky_bg():
    bg = starfield(9, 700, (0.03, 0.04, 0.08))
    r = np.random.default_rng(4)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(14):  # 星官连线（示意）
        x, y = r.uniform(100, W - 100), r.uniform(80, H - 80)
        pts = [(x, y)]
        for _ in range(int(r.integers(3, 6))):
            x += r.normal(0, 60)
            y += r.normal(0, 50)
            pts.append((x, y))
        d.line(pts, fill=(200, 180, 130, 70), width=1)
        for p in pts:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], outline=(220, 200, 150, 140))
    a = np.asarray(im, np.float32) / 255
    return bg * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]


SKY = _sky_bg()


def armillary(t_rot, cx=1240, cy=520, R=330):
    """张衡浑天仪：固定的地平环、子午环，绕极轴转动的赤道环、黄道环与窥管。"""
    im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cam_m = _rot("x", np.radians(-16)) @ _rot("y", np.radians(28))
    tilt = _rot("z", np.radians(35))            # 极轴倾角
    spin = _rot("y", t_rot)
    rings = [  # (半径比例, 局部旋转, 线宽, 刻度)
        (1.00, np.eye(3), 9, True),                                   # 地平环（水平）
        (1.00, _rot("x", np.pi / 2), 9, True),                        # 子午环（竖直）
        (0.86, tilt @ spin @ np.eye(3), 7, True),                     # 赤道环
        (0.86, tilt @ spin @ _rot("z", np.radians(23.5)), 6, False),  # 黄道环
        (0.72, tilt @ spin @ _rot("x", np.pi / 2), 6, True),          # 四游环
    ]

    def proj(p):
        q = cam_m @ p
        return (cx + q[0] * R) * S, (cy - q[1] * R) * S, q[2]

    segs = []
    for rr, m, wd, ticks in rings:
        th = np.linspace(0, 2 * np.pi, 181)
        pts = [proj(m @ np.array([np.cos(a) * rr, 0, np.sin(a) * rr])) for a in th]
        for i in range(180):
            (x0, y0, z0), (x1, y1, z1) = pts[i], pts[i + 1]
            segs.append(((z0 + z1) / 2, "line", (x0, y0, x1, y1), wd))
        if ticks:
            for a in np.linspace(0, 2 * np.pi, 49)[:-1]:
                p0 = proj(m @ np.array([np.cos(a) * rr * 0.95, 0, np.sin(a) * rr * 0.95]))
                p1 = proj(m @ np.array([np.cos(a) * rr * 1.03, 0, np.sin(a) * rr * 1.03]))
                segs.append(((p0[2] + p1[2]) / 2, "tick", (p0[0], p0[1], p1[0], p1[1]), 2))
    # 极轴与窥管
    ax_m = tilt @ spin
    a0, a1 = proj(ax_m @ np.array([0, -1.1, 0])), proj(ax_m @ np.array([0, 1.1, 0]))
    segs.append(((a0[2] + a1[2]) / 2, "line", (a0[0], a0[1], a1[0], a1[1]), 5))
    tube = tilt @ spin @ _rot("z", np.radians(50))
    t0, t1 = proj(tube @ np.array([-0.68, 0, 0])), proj(tube @ np.array([0.68, 0, 0]))
    segs.append(((t0[2] + t1[2]) / 2, "line", (t0[0], t0[1], t1[0], t1[1]), 11))
    segs.sort(key=lambda s: s[0])
    for z, kind, (x0, y0, x1, y1), wd in segs:
        k = 0.55 + 0.45 * np.clip((z + 1) / 2, 0, 1)
        col = tuple(int(v * k) for v in BRONZE) + (255,)
        d.line([(x0, y0), (x1, y1)], fill=col, width=wd * S)
        if kind == "line" and wd >= 6 and z > 0.2:
            d.line([(x0, y0 - S), (x1, y1 - S)], fill=(250, 220, 160, 120), width=S)
    # 底座：四根立柱与十字座
    base_y = cy + R * 1.18
    for dx in (-0.75, -0.25, 0.25, 0.75):
        d.line([((cx + dx * R) * S, (cy + R * 0.55) * S), ((cx + dx * R * 1.15) * S, base_y * S)],
               fill=(150, 112, 62, 255), width=8 * S)
    d.rectangle([(cx - R * 1.05) * S, base_y * S, (cx + R * 1.05) * S, (base_y + 16) * S], fill=(120, 90, 50, 255))
    im = im.resize((W, H), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    return a[..., :3], a[..., 3:]


def s_zhangheng(c):
    # 每拍转一格：拍内用缓动走完 7.5°
    step = np.floor(c.lb) + ease_io(c.lb - np.floor(c.lb))
    rgb, a = armillary(np.radians(7.5) * step + 0.3)
    z = lerp(1.0, 1.08, ease_io(c.k))
    f = SKY * (1 - a) + rgb * a
    f = f + glow(1240, 520, 420, (0.25, 0.18, 0.08), 0.25)
    f, _ = place(f, 1240, 540, z, 0, 1240, 540)
    caption(f, c, 0.5, [("张衡", "big"), ("让仪器跟着天空一起转", "mid"), ("公元 117 年 · 水运浑天仪", "small")])
    return post(f, c.fi, grain=0.03)


# ---------------- 伽利略 ----------------

def _scope(frame, content, cx, cy, r):
    m = circle_mask(cx, cy, r, 2.0)
    frame = frame * (1 - m) + content * m
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rim = np.clip(1 - np.abs(d - r - 10) / 12, 0, 1)[..., None]
    frame = frame * (1 - rim) + np.array([0.62, 0.48, 0.26], np.float32) * rim * (0.8 + 0.2 * (yy < cy)[..., None])
    vign = np.clip((d - r * 0.6) / (r * 0.45), 0, 1)[..., None] * m
    return frame * (1 - vign * 0.55)


def _jupiter(lb):
    """伽利略手稿风格的木星与四颗卫星：卫星位置随拍子变化。"""
    f = PAPER.copy()
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = 1250, 540
    d.ellipse([cx - 34, cy - 34, cx + 34, cy + 34], outline=(40, 28, 18, 255), width=4)
    for k in (-12, 0, 12):
        d.line([(cx - 30, cy + k), (cx + 30, cy + k)], fill=(40, 28, 18, 200), width=2)
    periods = [1.77, 3.55, 7.15, 16.7]
    radii = [90, 140, 210, 300]
    for p, r in zip(periods, radii):
        x = cx + r * np.cos(lb * 1.4 / p * 2 * np.pi / 2 + r)
        d.text((x, cy), "*", fill=(40, 28, 18, 255), anchor="mm",
               font=__import__("engine.core", fromlist=["font"]).font("serif", 54))
    a = np.asarray(im, np.float32) / 255
    return f * (1 - a[..., 3:]) + a[..., :3] * a[..., 3:]


def s_galileo(c):
    f = PAPER_DARK.copy()
    wood = image("PD-01", "02", 2400)
    luna = image("NA-03", "01", 2400)
    cx, cy, r = 1250, 540, 360
    sw = 2 * r / (0.9 * wood.shape[1])
    w_rgb, _ = place(wood, 0.5 * wood.shape[1], 0.265 * wood.shape[0], sw * lerp(1.0, 1.05, c.k), 0, cx, cy)
    sl = r / (0.47 * luna.shape[0])
    l_rgb, _ = place(luna, 0.5 * luna.shape[1], 0.5 * luna.shape[0], sl * lerp(1.0, 1.05, c.k), 0, cx, cy)
    mix = ease_io(clamp01((c.lb - 4.0) / 1.5))
    content = w_rgb * (1 - mix) + l_rgb * mix
    if c.lb >= 8.0:
        jm = ease_io(clamp01((c.lb - 8.0) / 0.8))
        content = content * (1 - jm) + _jupiter(c.lb - 8.0) * jm
    f = _scope(f, content, cx, cy, r)
    caption(f, c, 0.5, [("伽利略", "big"), ("月亮并不完美，它坑坑洼洼", "mid"), ("1609 年 · 第一次用望远镜看月亮", "small")],
            out=7.6)
    caption(f, c, 8.4, [("木星", "big"), ("1610 年，他看见四颗卫星绕着木星转", "mid"), ("不是所有天体都绕着地球转", "small")])
    return post(f, c.fi, grain=0.035)


# ---------------- 开普勒 ----------------

A_ELL, E_ELL = 430, 0.58
B_ELL = A_ELL * np.sqrt(1 - E_ELL ** 2)
C_ELL = A_ELL * E_ELL
ECX, ECY = 1220, 560


def _kepler_pos(M_):
    E = M_
    for _ in range(12):
        E = E - (E - E_ELL * np.sin(E) - M_) / (1 - E_ELL * np.cos(E))
    return ECX + A_ELL * np.cos(E), ECY + B_ELL * np.sin(E)


def _orbit_layer(draw_k, planet_M, sweeps):
    im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ink = (44, 30, 18, 255)
    sun = (ECX + C_ELL, ECY)
    for M0, M1 in sweeps:  # 等时扫过的面积
        pts = [sun] + [_kepler_pos(m) for m in np.linspace(M0, M1, 30)]
        d.polygon([(x * S, y * S) for x, y in pts], fill=(150, 110, 60, 70), outline=(90, 60, 30, 160))
    n = max(2, int(200 * draw_k))
    th = np.linspace(0, 2 * np.pi * draw_k, n)
    d.line([((ECX + A_ELL * np.cos(a)) * S, (ECY + B_ELL * np.sin(a)) * S) for a in th], fill=ink, width=3 * S)
    d.line([((ECX - A_ELL - 40) * S, ECY * S), ((ECX + A_ELL + 40) * S, ECY * S)], fill=(44, 30, 18, 90), width=S)
    sx, sy = sun
    d.ellipse([(sx - 20) * S, (sy - 20) * S, (sx + 20) * S, (sy + 20) * S], fill=(200, 140, 50, 255), outline=ink, width=2 * S)
    for k in range(12):
        a = k * np.pi / 6
        d.line([((sx + 26 * np.cos(a)) * S, (sy + 26 * np.sin(a)) * S), ((sx + 38 * np.cos(a)) * S, (sy + 38 * np.sin(a)) * S)],
               fill=ink, width=2 * S)
    if planet_M is not None:
        px, py = _kepler_pos(planet_M)
        d.ellipse([(px - 12) * S, (py - 12) * S, (px + 12) * S, (py + 12) * S], fill=ink)
    im = im.resize((W, H), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    return a[..., :3], a[..., 3:]


def s_kepler(c):
    if c.lb < 3.5:
        src = duotone(image("PD-02", "01", 2400), (0.12, 0.08, 0.05), (0.93, 0.87, 0.74), contrast=1.1)
        f = ken(c, src, (0.5, 0.42, 0.85), (0.5, 0.35, 1.05), t1=3.5, mode="fit", bg=PAPER_DARK, shadow=0.6)
        f = f * (1 - clamp01((c.lb - 3.1) / 0.4)) + PAPER * clamp01((c.lb - 3.1) / 0.4)
    else:
        lb = c.lb - 3.5
        draw_k = ease_out(clamp01(lb / 2.0), 2)
        M_ = (lb - 1.5) / 4.5 * 2 * np.pi if lb > 1.5 else None
        sweeps = []
        if lb > 3.0:
            sweeps.append((-0.35, 0.35))
        if lb > 4.0:
            sweeps.append((np.pi - 0.35, np.pi + 0.35))
        rgb, a = _orbit_layer(draw_k, M_, sweeps)
        f = PAPER * (1 - a) + rgb * a
    color = CREAM if c.lb < 3.5 else INK
    caption(f, c, 0.6, [("开普勒", "big"), ("行星走的不是正圆", "mid"), ("1609 / 1619 年 · 行星运动定律", "small")],
            color=color, scrim=0.6 if c.lb < 3.5 else 0.0)
    return post(f, c.fi, grain=0.03)


# ---------------- 牛顿 ----------------

NR, NMX, NMY = 300, 1250, 650       # 地球半径与圆心
MOUNT = 70


def _newton_static():
    im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ink = (44, 30, 18, 255)
    d.ellipse([(NMX - NR) * S, (NMY - NR) * S, (NMX + NR) * S, (NMY + NR) * S], outline=ink, width=3 * S)
    mask = Image.new("L", (W * S, H * S), 0)
    ImageDraw.Draw(mask).ellipse([(NMX - NR) * S, (NMY - NR) * S, (NMX + NR) * S, (NMY + NR) * S], fill=255)
    hatch = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hatch)
    for y in range(NMY - NR, NMY + NR, 7):
        hd.line([((NMX - NR) * S, y * S), ((NMX + NR) * S, y * S)], fill=(44, 30, 18, 90), width=S)
    im.paste(hatch, (0, 0), mask)
    top = NMY - NR
    d.polygon([((NMX - 80) * S, (top + 10) * S), (NMX * S, (top - MOUNT) * S), ((NMX + 80) * S, (top + 10) * S)],
              fill=(120, 96, 66, 255), outline=ink)
    d.rectangle([(NMX - 4) * S, (top - MOUNT - 10) * S, (NMX + 30) * S, (top - MOUNT - 2) * S], fill=ink)  # 炮
    im = im.resize((W, H), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    return a[..., :3], a[..., 3:]


NSTAT = _newton_static()


def _traj(v_frac, steps=4000):
    r0 = NR + MOUNT + 6
    gm = 1.0
    vc = np.sqrt(gm / r0)
    pos = np.array([0.0, r0])
    vel = np.array([vc * v_frac, 0.0])
    pts = [pos.copy()]
    T = 2 * np.pi * np.sqrt(r0 ** 3 / gm)
    dt = T / steps
    for _ in range(steps):
        rr = np.linalg.norm(pos)
        acc = -gm * pos / rr ** 3
        vel = vel + acc * dt
        pos = pos + vel * dt
        pts.append(pos.copy())
        if np.linalg.norm(pos) < NR:
            break
    return np.array([(NMX + p[0], NMY - p[1]) for p in pts])


TRAJ = [_traj(v) for v in (0.6, 0.78, 0.9, 1.0)]


def s_newton(c):
    lb = c.lb
    if lb < 3.0:
        src = duotone(image("PD-03", "01", 2400), (0.12, 0.08, 0.05), (0.93, 0.87, 0.74), contrast=1.1)
        f = ken(c, src, (0.5, 0.45, 0.9), (0.37, 0.36, 2.2), t1=3.0, mode="fit", bg=PAPER_DARK, shadow=0.5)
        f = f * (1 - clamp01((lb - 2.6) / 0.4)) + PAPER * clamp01((lb - 2.6) / 0.4)
    else:
        rgb, a = NSTAT
        f = PAPER * (1 - a) + rgb * a
        im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for i, tr in enumerate(TRAJ):
            b0 = 3.2 + i * 1.6
            if lb < b0:
                continue
            dur = 1.4 if i < 3 else 3.0
            k = ease_out(clamp01((lb - b0) / dur), 1.6) if i < 3 else clamp01((lb - b0) / dur)
            n = max(2, int(len(tr) * k))
            col = (44, 30, 18, 255) if i < 3 else (150, 40, 26, 255)
            seg = tr[:n]
            for j in range(0, n - 1, 6):  # 虚线
                d.line([(seg[j][0] * S, seg[j][1] * S), (seg[min(j + 3, n - 1)][0] * S, seg[min(j + 3, n - 1)][1] * S)],
                       fill=col, width=3 * S)
            hx, hy = seg[-1]
            d.ellipse([(hx - 8) * S, (hy - 8) * S, (hx + 8) * S, (hy + 8) * S], fill=col)
        im = im.resize((W, H), Image.LANCZOS)
        a2 = np.asarray(im, np.float32) / 255
        f = f * (1 - a2[..., 3:]) + a2[..., :3] * a2[..., 3:]
    iss_k = ease_io(clamp01((lb - 11.0) / 1.2))
    if iss_k > 0:
        v = video(c, ISS, c.since(11.0) * 0.8 + 1.0, zoom=1.05)
        f = f * (1 - iss_k) + v * iss_k
    if lb < 11.0:
        caption(f, c, 0.6, [("牛顿", "big"), ("只要足够快，就永远不会落地", "mid"), ("1687 年 · 《自然哲学的数学原理》", "small")],
                color=CREAM if lb < 3.0 else INK, scrim=0.6 if lb < 3.0 else 0.0)
    else:
        caption(f, c, 11.6, [("这就是轨道", "big")], y=H - 330)
    return post(f, c.fi, grain=0.03)


SHOTS = [Shot(8, s_card, name="2-01"), Shot(12, s_zhangheng, name="2-02"), Shot(12, s_galileo, name="2-03"),
         Shot(10, s_kepler, name="2-04"), Shot(15, s_newton, name="2-05")]

CHORDS = {"Dm": [50, 53, 57], "Bb": [46, 50, 53], "F": [45, 48, 53], "C": [48, 52, 55], "Gm": [43, 50, 55],
          "A": [45, 49, 52]}


def music(sc, ch):
    T, t = ch.T, ch.t
    sc.add("sfx", M.real("FX/FX-05/FX-05_01_paper_turn_pixabay.mp3", fade=0.02), T(0.2), 0.3)
    sc.add("sfx", M.stamp(), T(2.5), 0.6)
    prog = ["Dm", "Bb", "F", "C", "Dm", "Gm", "A", "A"]
    total = ch.total_beats
    # 拨奏固定音型（八分音符）+ 弦乐长音，从张衡开始
    for bar in range(int(8 // 4), int(total // 4)):
        chn = prog[bar % len(prog)]
        b0 = bar * 4
        notes = CHORDS[chn]
        sc.add("bgm", M.pad([n + 12 for n in notes], t(b0 + 4) - t(b0) + 0.2, 1600, 0.3), T(b0), 0.3)
        sc.add("bgm", M.strings_long(notes[0] - 12, t(b0 + 4) - t(b0)), T(b0), 0.3)
        pat = [0, 2, 1, 2, 0, 2, 1, 2]
        for i in range(8):
            m = notes[pat[i]] + (12 if i % 4 == 3 else 0)
            sc.add("bgm", M.pluck(m, 0.8, 1.1, 0.35), T(b0 + i * 0.5), 0.3, pan=-0.2 + 0.4 * (i % 2))
    # 张衡：每拍一声齿轮
    b_z = ch.shot_start(1)
    for b in np.arange(b_z, b_z + 12):
        sc.add("sfx", M.tick() * 2.5, T(b), 0.35)
    # 伽利略：叠化成真实月面时的钟声
    sc.add("bgm", M.bell(81, 3.0), T(ch.shot_start(2) + 4.5), 0.25)
    # 开普勒：竖琴般的上行
    for k, m in enumerate([62, 65, 69, 72, 74, 77, 81]):
        sc.add("bgm", M.pluck(m, 1.6, 1.3, 0.8), T(ch.shot_start(3) + 3.5 + k * 0.25), 0.22)
    # 牛顿：每发炮弹一声闷响，最后一发绕地一周 → 合唱与铜管铺开
    b_n = ch.shot_start(4)
    for i in range(4):
        sc.add("sfx", M.boom(1.2, 70, 40, 0.15), T(b_n + 3.2 + i * 1.6), 0.25)
    sc.add("bgm", M.riser(t(b_n + 11) - t(b_n + 8)), T(b_n + 8), 0.2)
    sc.add("bgm", M.choir([62, 65, 69, 74], t(b_n + 15) - t(b_n + 11) + 1.0, 0.6), T(b_n + 11), 0.45)
    sc.add("bgm", M.timpani(38, 2.0, 1.0), T(b_n + 11), 0.6)
    sc.add("bgm", M.brass_stab([62, 65, 69, 74], 1.2), T(b_n + 11), 0.4)


CHAPTER = Chapter("gaze", "仰望", 84, SHOTS, clips=[ISS], music=music)
