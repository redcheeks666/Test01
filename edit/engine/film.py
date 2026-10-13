"""全片时间轴与渲染。

每章是一个 Chapter：自己的速度（可渐变）和镜头清单。镜头函数签名为 fn(c, **kw) → frame，
c 是 Ctx，提供镜头内拍号、秒数等。全片按章节顺序首尾相接。
"""
import multiprocessing as mp
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .core import FPS, W, H, ROOT

OUT = ROOT / "edit" / "out"


@dataclass
class Shot:
    beats: float
    fn: object
    kw: dict = field(default_factory=dict)
    name: str = ""


class Ctx:
    __slots__ = ("ch", "fi", "sec", "b", "b0", "beats", "lb", "shot")

    def t(self, lb):
        """镜头内拍号 → 章节内秒数。"""
        return self.ch.t(self.b0 + lb)

    def since(self, lb):
        """距镜头内拍号 lb 已过去的秒数（可为负）。"""
        return self.sec - self.ch.t(self.b0 + lb)

    @property
    def lsec(self):
        return self.sec - self.ch.t(self.b0)

    @property
    def dur(self):
        return self.ch.t(self.b0 + self.beats) - self.ch.t(self.b0)

    @property
    def k(self):
        return min(max(self.lb / self.beats, 0.0), 1.0)


class Chapter:
    def __init__(self, key, title, bpm0, shots, bpm1=None, ramp_beats=0, overlay=None, clips=(), music=None):
        self.key, self.title = key, title
        self.bpm0, self.bpm1 = bpm0, bpm1 if bpm1 is not None else bpm0
        self.ramp = ramp_beats
        self.shots = shots
        self.overlay = overlay
        self.clips = list(clips)
        self.music = music
        self.starts = np.concatenate([[0], np.cumsum([s.beats for s in shots])])
        self.total_beats = float(self.starts[-1])
        step = 1 / 64
        self._grid = np.arange(0, self.total_beats + 8, step)
        bpm = self.bpm0 + (self.bpm1 - self.bpm0) * np.clip(self._grid / max(self.ramp, 1e-9), 0, 1) \
            if self.ramp else np.full_like(self._grid, self.bpm0)
        self._times = np.concatenate([[0], np.cumsum(60.0 / bpm[:-1] * step)])
        self.duration = self.t(self.total_beats)
        self.T0 = 0.0

    def t(self, b):
        return float(np.interp(b, self._grid, self._times))

    def T(self, b):
        """章节内拍号 → 全片秒数。"""
        return self.T0 + self.t(b)

    def beat_at(self, sec):
        return float(np.interp(sec, self._times, self._grid))

    def shot_start(self, i):
        return float(self.starts[i])

    def frame(self, fi):
        sec = fi / FPS - self.T0
        b = self.beat_at(sec)
        i = int(np.searchsorted(self.starts, b, side="right") - 1)
        i = min(max(i, 0), len(self.shots) - 1)
        s = self.shots[i]
        c = Ctx()
        c.ch, c.fi, c.sec, c.b, c.b0, c.beats, c.lb, c.shot = self, fi, sec, b, float(self.starts[i]), s.beats, \
            b - float(self.starts[i]), s
        img = s.fn(c, **s.kw)
        if self.overlay is not None:
            img = self.overlay(img, c)
        return img


class Film:
    def __init__(self, chapters):
        self.chapters = chapters
        t = 0.0
        for ch in chapters:
            ch.T0 = t
            t += ch.duration
        self.duration = t

    def chapter(self, key):
        return next(c for c in self.chapters if c.key == key)

    def frame_range(self, ch):
        f0 = int(np.ceil(ch.T0 * FPS - 1e-6))
        f1 = int(np.ceil((ch.T0 + ch.duration) * FPS - 1e-6))
        return f0, f1


# ---------------- 渲染 ----------------

_FILM = None


def _worker(args):
    """在独立进程（spawn）里加载全片并渲染一段帧，避免 fork 继承 OpenCV 线程池导致死锁。"""
    key, f0, f1, part, rng = args
    cv2.setNumThreads(1)
    import sys
    sys.path.insert(0, str(ROOT / "edit"))
    from tianwen import load_film
    film = load_film()
    ch = film.chapter(key)
    if film.frame_range(ch) != rng:
        # 渲染途中改了别的章节的长度：主进程分的帧号和子进程读到的时间轴对不上，画面会整体错位
        raise RuntimeError(f"{key}: 时间轴在渲染途中变了（主进程 {rng}，子进程 {film.frame_range(ch)}），请重新渲染")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-g", "60",
           "-pix_fmt", "yuv420p", str(part)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        img = ch.frame(fi)
        p.stdin.write((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    return part


def render_chapter(film, key, workers=4, frames=None):
    """渲染一章到 edit/out/chapters/<key>.mp4（无音轨）。frames 给定时只输出这些帧的 jpg 做检查。"""
    global _FILM
    _FILM = film
    ch = film.chapter(key)
    for clip in ch.clips:
        clip.ensure()
    f0, f1 = film.frame_range(ch)
    d = OUT / "chapters"
    d.mkdir(parents=True, exist_ok=True)
    if frames is not None:
        from PIL import Image
        paths = []
        for fi in frames:
            fi = f0 + fi
            img = (np.clip(ch.frame(fi), 0, 1) * 255).astype(np.uint8)
            p = d / f"{key}_{fi - f0:05d}.jpg"
            Image.fromarray(img).save(p, quality=88)
            paths.append(p)
        return paths
    n = f1 - f0
    bounds = np.linspace(f0, f1, workers + 1).astype(int)
    parts = [(key, int(a), int(b), d / f"{key}_part{i}.mp4", (f0, f1)) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))
             if b > a]
    with mp.get_context("spawn").Pool(len(parts)) as pool:
        pool.map(_worker, parts)
    lst = d / f"{key}_parts.txt"
    lst.write_text("".join(f"file '{p[3].name}'\n" for p in parts))
    out = d / f"{key}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)
    for p in parts:
        Path(p[3]).unlink()
    lst.unlink()
    return out, n
