"""素材：图片读取缓存、视频片段抽帧与变速。

视频片段先用 ffmpeg 把需要的时间段抽成 JPEG 序列缓存到 edit/.cache/clips/，渲染进程按源时间取帧，
可在相邻两帧之间做线性混合（慢放时更顺）。
"""
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .core import W, H, ROOT

import sys
sys.path.insert(0, str(ROOT / "edit"))
from assets import load_image, resolve, find  # noqa: E402

CLIP_CACHE = ROOT / "edit" / ".cache" / "clips"


@lru_cache(maxsize=64)
def image(asset_id, candidate="01", max_edge=3000):
    return load_image(asset_id, candidate, max_edge)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return s["width"], s["height"], fps


class Clip:
    """视频片段。src 为 assets 相对路径（或 Path），t_in/t_out 为源时间（秒），
    crop 为归一化裁切框 (x0, y0, x1, y1)，size 为缓存尺寸（留出推镜余量可大于 1920×1080）。"""

    def __init__(self, name, src, t_in, t_out, crop=None, size=(2304, 1296)):
        self.name, self.src, self.t_in, self.t_out = name, src, t_in, t_out
        self.crop = crop
        self.size = size
        self.dir = CLIP_CACHE / name
        self._fps = None

    def path(self):
        if isinstance(self.src, Path):
            return self.src
        return resolve(self.src)

    def ensure(self):
        meta = self.dir / "meta.json"
        if meta.exists():
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.path()
        w, h, fps = probe(p)
        vf = []
        if self.crop:
            x0, y0, x1, y1 = self.crop
            vf.append(f"crop={int((x1 - x0) * w)}:{int((y1 - y0) * h)}:{int(x0 * w)}:{int(y0 * h)}")
        vf.append(f"scale={self.size[0]}:{self.size[1]}:flags=lanczos")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{self.t_in:.3f}", "-to", f"{self.t_out:.3f}",
                        "-i", str(p), "-vf", ",".join(vf), "-q:v", "3", str(self.dir / "%05d.jpg")], check=True)
        n = len(list(self.dir.glob("*.jpg")))
        meta.write_text(json.dumps({"fps": fps, "frames": n}))

    @property
    def fps(self):
        if self._fps is None:
            m = json.loads((self.dir / "meta.json").read_text())
            self._fps, self.n = m["fps"], m["frames"]
        return self._fps

    @property
    def duration(self):
        return self.fps and self.n / self.fps

    @lru_cache(maxsize=8)
    def _frame(self, i):
        i = int(np.clip(i, 0, self.n - 1))
        img = cv2.imread(str(self.dir / f"{i + 1:05d}.jpg"), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def at(self, src_t, blend=True):
        """取源时间 src_t（相对片段起点，秒）的画面。"""
        f = src_t * self.fps
        i0 = int(np.floor(f))
        if not blend:
            return self._frame(int(round(f)))
        k = f - i0
        a = self._frame(i0)
        if k < 0.02 or i0 + 1 >= self.n:
            return a
        return a * (1 - k) + self._frame(i0 + 1) * k


def retime(keys, start=0.0, step=1 / 240):
    """变速曲线。keys = [(镜头内秒数, 速度)]，相邻关键点之间平滑过渡。返回 f(镜头内秒数) → 源时间。"""
    keys = sorted(keys)
    t_end = keys[-1][0] + 30
    ts = np.arange(0, t_end, step)
    sp = np.empty_like(ts)
    for i, t in enumerate(ts):
        if t <= keys[0][0]:
            sp[i] = keys[0][1]
        elif t >= keys[-1][0]:
            sp[i] = keys[-1][1]
        else:
            for (t0, s0), (t1, s1) in zip(keys, keys[1:]):
                if t0 <= t <= t1:
                    u = (t - t0) / max(t1 - t0, 1e-6)
                    u = u * u * (3 - 2 * u)
                    sp[i] = s0 + (s1 - s0) * u
                    break
    src = start + np.concatenate([[0], np.cumsum(sp[:-1] * step)])
    return lambda t: float(np.interp(t, ts, src))
