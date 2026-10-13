"""万户：剪影插画。巨大的月亮作背景，万户坐在官帽椅上，椅背、椅下绑着 47 支火箭，手牵两只风筝。

剪影比线描更好认：椅子、人物、火箭的轮廓在月亮前一眼就能看出来。
输出：天空（含月亮）RGB、剪影 RGBA、引信路径。坐标系为 1920×1080。
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw

W, H, S = 1920, 1080, 2
SIL = (24, 17, 12, 255)
EDGE = (92, 66, 44, 255)
TIE = (120, 88, 58, 255)


def _fbm(h, w, octaves, seed):
    r = np.random.default_rng(seed)
    out, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    for o in range(octaves):
        s = 2 ** (o + 2)
        small = r.random((s * h // w + 2, s + 2)).astype(np.float32)
        out += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp
        tot += amp
        amp *= 0.5
    return out / tot


MOON_C, MOON_R = (1262, 430), 236


def sky():
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top, mid, hor = np.array([0.06, 0.06, 0.09]), np.array([0.19, 0.14, 0.15]), np.array([0.74, 0.47, 0.26])
    g = np.where(y < 0.55, top + (mid - top) * (y / 0.55), mid + (hor - mid) * np.clip((y - 0.55) / 0.32, 0, 1))
    img = np.broadcast_to(g, (H, W, 3)).copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # 星星
    r = np.random.default_rng(8)
    for _ in range(260):
        sx, sy = r.uniform(0, W), r.uniform(0, H * 0.6)
        b = r.uniform(0.2, 0.8) * (1 - sy / (H * 0.6))
        cv2.circle(img, (int(sx), int(sy)), 1, (b, b * 0.95, b * 0.85), -1)
    # 月晕
    d = np.sqrt((xx - MOON_C[0]) ** 2 + (yy - MOON_C[1]) ** 2)
    img += np.exp(-((d - MOON_R) / 150).clip(0) ** 2)[..., None] * np.array([0.30, 0.24, 0.17]) * 0.55
    # 月面：亮底 + 月海暗斑 + 边缘变暗
    tex = _fbm(H, W, 6, 41)
    maria = np.clip((_fbm(H, W, 4, 77) - 0.52) * 4, 0, 1)
    rr = np.clip(d / MOON_R, 0, 1)
    moon = np.array([0.97, 0.91, 0.76]) * ((0.86 + 0.22 * tex) * (1 - 0.32 * maria) * (1 - 0.28 * rr ** 2))[..., None]
    a = np.clip(MOON_R + 0.8 - d, 0, 1)[..., None]
    img = img * (1 - a) + moon * a
    return np.clip(img, 0, 1).astype(np.float32)


def _p(pts):
    return [(x * S, y * S) for x, y in pts]


def _rocket(d, x, yb, L, w=13):
    nose = w * 1.6
    d.rectangle(_p([(x - w / 2, yb - L + nose), (x + w / 2, yb)]), fill=SIL, outline=EDGE, width=S)
    d.polygon(_p([(x - w / 2, yb - L + nose), (x + w / 2, yb - L + nose), (x, yb - L)]), fill=SIL, outline=EDGE)
    d.polygon(_p([(x - w / 2 + 2, yb), (x + w / 2 - 2, yb), (x + w / 2 + 1, yb + 7), (x - w / 2 - 1, yb + 7)]), fill=SIL)
    for k in (0.35, 0.7):  # 绑绳
        yy = yb - (L - nose) * k
        d.line(_p([(x - w / 2, yy), (x + w / 2, yy)]), fill=TIE, width=2 * S)


def silhouette():
    im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    rockets_bottom = []

    def cluster(xs, yb, L):
        for x in xs:
            _rocket(d, x, yb, L)
        rockets_bottom.append((float(np.mean(xs)), yb + 6))

    # 远处的山丘与发射台
    hill = [(0, 1080), (0, 990), (300, 965), (620, 940), (900, 922), (1250, 902), (1600, 925), (1920, 958), (1920, 1080)]
    d.polygon(_p(hill), fill=SIL)
    d.rectangle(_p([(1070, 862), (1500, 880)]), fill=SIL, outline=EDGE, width=S)
    for x in (1090, 1280, 1480):
        d.rectangle(_p([(x - 6, 880), (x + 6, 905)]), fill=SIL)

    # 椅背后方与上方、椅下、右侧的火箭（先画，在人物和椅子后面）
    cluster([1340 + i * 15 for i in range(7)], 598, 118)      # 椅背后上排 7
    cluster([1318 + i * 14 for i in range(7)], 468, 105)      # 搭脑上方 7
    cluster([1340 + i * 15 for i in range(7)], 712, 125)      # 椅背后下排 7
    cluster([1445 + i * 14 for i in range(4)], 728, 118)      # 最右上 4
    cluster([1445 + i * 14 for i in range(4)], 860, 128)      # 最右下 4
    cluster([1330 + i * 15 for i in range(7)], 860, 135)      # 椅后下方 7
    cluster([1172 + i * 12 for i in range(11)], 856, 118)     # 椅下 11  → 共 47

    # 官帽椅（侧面）
    d.rectangle(_p([(1145, 700), (1330, 714)]), fill=SIL, outline=EDGE, width=S)           # 座面
    d.rectangle(_p([(1150, 714), (1162, 862)]), fill=SIL, outline=EDGE, width=S)           # 前腿
    d.rectangle(_p([(1302, 714), (1314, 862)]), fill=SIL, outline=EDGE, width=S)           # 后腿
    d.rectangle(_p([(1150, 818), (1314, 826)]), fill=SIL, outline=EDGE, width=S)           # 脚踏
    d.polygon(_p([(1304, 706), (1318, 706), (1342, 478), (1328, 478)]), fill=SIL, outline=EDGE)  # 后腿上延为椅背立柱
    d.polygon(_p([(1306, 474), (1388, 460), (1400, 452), (1398, 466), (1312, 484)]), fill=SIL, outline=EDGE)  # 搭脑（两端上翘）
    d.polygon(_p([(1150, 700), (1160, 700), (1160, 636), (1150, 636)]), fill=SIL, outline=EDGE)  # 扶手前柱
    d.polygon(_p([(1148, 628), (1322, 616), (1322, 626), (1148, 638)]), fill=SIL, outline=EDGE)  # 扶手

    # 万户：长袍坐姿，面向左上
    robe = [(1252, 552), (1236, 600), (1232, 650), (1180, 662), (1130, 676), (1120, 700), (1126, 812),
            (1178, 816), (1182, 714), (1300, 706), (1318, 660), (1312, 562), (1286, 548)]
    d.polygon(_p(robe), fill=SIL, outline=EDGE)
    d.polygon(_p([(1116, 812), (1152, 812), (1156, 826), (1110, 826)]), fill=SIL)              # 鞋
    sleeve = [(1282, 566), (1302, 592), (1270, 652), (1240, 656), (1198, 628), (1186, 606), (1208, 600),
              (1250, 626), (1268, 588)]
    d.polygon(_p(sleeve), fill=SIL, outline=EDGE)
    d.ellipse(_p([(1240, 500), (1296, 556)]), fill=SIL, outline=EDGE, width=S)                # 头
    d.polygon(_p([(1242, 520), (1232, 532), (1242, 536)]), fill=SIL)                           # 鼻
    d.chord(_p([(1238, 478), (1298, 530)]), 180, 360, fill=SIL, outline=EDGE)                  # 乌纱帽
    d.rectangle(_p([(1238, 500), (1298, 512)]), fill=SIL)
    d.ellipse(_p([(1290, 494), (1352, 506)]), fill=SIL, outline=EDGE)                          # 帽翅
    d.ellipse(_p([(1214, 496), (1244, 506)]), fill=SIL, outline=EDGE)

    # 两只风筝
    hand = (1190, 610)
    for (kx, ky, ks) in [(930, 300, 84), (1050, 190, 66)]:
        kite = [(kx, ky - ks), (kx + ks * 0.62, ky), (kx, ky + ks * 0.9), (kx - ks * 0.62, ky)]
        d.polygon(_p(kite), fill=SIL, outline=EDGE)
        d.line(_p([(kx, ky - ks), (kx, ky + ks * 0.9)]), fill=EDGE, width=S)
        d.line(_p([(kx - ks * 0.62, ky), (kx + ks * 0.62, ky)]), fill=EDGE, width=S)
        d.line(_p([(kx, ky + ks * 0.9), hand]), fill=(60, 44, 30, 200), width=S)
        for i in range(4):  # 尾巴上的布结
            tx, ty = kx - 10 - i * 14, ky + ks * 0.9 + 18 + i * 20
            d.polygon(_p([(tx - 7, ty - 5), (tx + 7, ty + 5), (tx + 7, ty - 5), (tx - 7, ty + 5)]), fill=SIL)

    im = im.resize((W, H), Image.LANCZOS)
    arr = np.asarray(im, np.float32) / 255.0
    fuse_main = [(860, 930), (980, 916), (1100, 884), (1240, 878)]
    return arr[..., :3], arr[..., 3:], fuse_main, rockets_bottom


def build():
    rgb, a, fuse_main, branches = silhouette()
    return {"sky": sky(), "rgb": rgb, "a": a, "fuse": fuse_main, "branches": branches}
