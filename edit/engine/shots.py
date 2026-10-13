"""通用镜头：照片运镜、视频、分屏、纪念镜头。"""
import cv2
import numpy as np

from .core import (W, H, FPS, CREAM, PAPER_DARK, cam, place, comp, cover_scale, fit_scale, lerp, ease_io,
                   clamp01, draw_text, post, motion_blur, text_width)


def ken(c, src, a, b, ease=ease_io, t0=0.0, t1=None, mode="cover", bg=None, shadow=0.0, sx=W / 2, sy=H / 2,
        out_w=W, out_h=H):
    """照片运镜。a、b = (cx, cy, zoom, rot)：cx、cy 为源图归一化坐标，zoom 相对铺满（cover）或完整显示（fit）的倍数。
    t0、t1 为运动的起止拍（镜头内），默认整个镜头。"""
    t1 = c.beats if t1 is None else t1
    k = ease(clamp01((c.lb - t0) / max(t1 - t0, 1e-6)))
    h, w = src.shape[:2]
    base = cover_scale(src, out_w, out_h) if mode == "cover" else fit_scale(src, out_w, out_h)
    cx, cy = lerp(a[0], b[0], k) * w, lerp(a[1], b[1], k) * h
    z = a[2] * (b[2] / a[2]) ** k
    rot = lerp(a[3] if len(a) > 3 else 0, b[3] if len(b) > 3 else 0, k)
    if mode == "cover" and bg is None:
        m = cv2.getRotationMatrix2D((float(cx), float(cy)), rot, base * z)
        m[0, 2] += sx - cx
        m[1, 2] += sy - cy
        return cv2.warpAffine(src, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    rgb, al = place(src, cx, cy, base * z, rot, sx, sy, out_w, out_h)
    bgi = (PAPER_DARK if bg is None else bg)
    bgi = bgi[:out_h, :out_w].copy() if bgi.shape[0] >= out_h else cv2.resize(bgi, (out_w, out_h))
    return comp(bgi, rgb, al, shadow)


def video(c, clip, src_t, grade=None, zoom=1.0, cx=0.5, cy=0.5, rot=0.0, blend=True):
    """视频画面：clip 缓存尺寸一般大于 1920×1080，zoom=1 时铺满，>1 时推近。"""
    img = clip.at(src_t, blend)
    h, w = img.shape[:2]
    s = cover_scale(img) * zoom
    m = cv2.getRotationMatrix2D((cx * w, cy * h), rot, s)
    m[0, 2] += W / 2 - cx * w
    m[1, 2] += H / 2 - cy * h
    out = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return grade(out) if grade else out


def split(left, right, line=1.0, color=(0.85, 0.75, 0.55), gap=4):
    """左右分屏：left、right 均为 960 宽的画面。line 控制中线从上往下画出的进度。"""
    frame = np.concatenate([left[:, : W // 2], right[:, : W // 2]], axis=1)
    ln = int(H * clamp01(line))
    frame[:ln, W // 2 - gap // 2:W // 2 + gap // 2] = np.array(color, np.float32)
    return frame


def memorial(c, img, title, names, sub="", fade_in=1.0, hold=None):
    """遇难事件：照片去饱和静止缓慢推近 → 压暗 → 黑场上出现名字。不加任何运动花样。"""
    hold = c.beats * 0.5 if hold is None else hold
    frame = np.zeros((H, W, 3), np.float32)
    if img is not None and c.lb < hold + 1.0:
        g = (img @ np.array([0.3, 0.59, 0.11], np.float32))[..., None].repeat(3, 2)
        f = ken(c, g, (0.5, 0.5, 1.0), (0.5, 0.5, 1.04))
        dim = clamp01((c.lb - hold) / 1.0)
        frame = f * (1 - dim) * clamp01(c.lsec / fade_in)
    if c.lb >= hold:
        a = clamp01((c.lb - hold - 0.3) * 1.5)
        draw_text(frame, title, "serif", 44, CREAM, W / 2, H / 2 - 120, a, anchor="c", tracking=6)
        for i, n in enumerate(names):
            ai = clamp01((c.lb - hold - 0.6 - i * 0.18) * 2)
            draw_text(frame, n, "serif", 30, (0.85, 0.82, 0.76), W / 2, H / 2 - 30 + i * 46, ai, anchor="c",
                      tracking=4, weight=400)
        if sub:
            draw_text(frame, sub, "serif", 26, (0.7, 0.68, 0.62), W / 2, H / 2 - 30 + len(names) * 46 + 30,
                      clamp01((c.lb - hold - 1.5) * 2), anchor="c")
    return post(frame, c.fi, grain=0.03, vignette=False)


def whip(frame, k, direction=1, max_px=900, blur=120):
    """甩镜：k 从 1 到 0 表示从画外甩入（direction=1 从右侧）。"""
    if k <= 0.01:
        return frame
    m = np.float32([[1, 0, direction * k * max_px], [0, 1, 0]])
    f = cv2.warpAffine(frame, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return motion_blur(f, blur * k, 0)


def framed(src, w, h, border=0.0, color=(0.9, 0.86, 0.78)):
    """把 src 缩放进 w×h 的框里（铺满裁切），可加细边框。"""
    s = cover_scale(src, w, h)
    rgb, _ = place(src, src.shape[1] / 2, src.shape[0] / 2, s, 0, w / 2, h / 2, w, h)
    if border:
        b = int(border)
        rgb[:b], rgb[-b:], rgb[:, :b], rgb[:, -b:] = color, color, color, color
    return rgb
