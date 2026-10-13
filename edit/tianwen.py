"""《天问》全片入口。

python3 tianwen.py frames <章节> <镜头内帧号...>   # 输出检查帧
python3 tianwen.py render <章节|all> [--workers 4]
python3 tianwen.py music                          # 生成全片配乐 out/score.wav
python3 tianwen.py assemble                       # 拼接各章 + 配乐 → out/tianwen_master.mp4
python3 tianwen.py info                           # 打印各章时长
python3 tianwen.py deliver                        # 上传版 1080p（90 MiB 分片 + sha256）与 <30MB 的 720p 分段预览
"""
import argparse
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine.film import Film, render_chapter, OUT  # noqa: E402
from engine.core import FPS  # noqa: E402

ORDER = ["prologue", "myth", "gaze", "pioneers", "race_a", "race_b", "return_", "epilogue"]
DELIVERY = HERE.parent / "delivery"
PART = 90 * 1024 * 1024                      # 与 assets/ 相同：每片 90 MiB，小于 95 MB
PREVIEW_GROUPS = [("prologue", "gaze"), ("pioneers", "race_a"), ("race_b", "return_"), ("epilogue", "epilogue")]
PREVIEW_BYTES = 28_000_000                   # 每段预览 < 30 MB


def load_film(only=None):
    chapters = []
    for key in ORDER:
        try:
            mod = importlib.import_module(f"chapters.{key}")
        except ModuleNotFoundError as e:
            if e.name == f"chapters.{key}":
                continue
            raise
        chapters.append(mod.CHAPTER)
    return Film(chapters)


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def deliver(film):
    """上传版：1080p H.264 High + AAC 320k，按 90 MiB 分片并附 sha256 入库；预览：按章节分段的 720p，每段 < 30 MB。"""
    master = OUT / "tianwen_master.mp4"
    DELIVERY.mkdir(exist_ok=True)
    final = DELIVERY / "tianwen_1080p.mp4"
    _ff("-i", master, "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", "12M", "-bufsize", "24M",
        "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p", "-g", "60",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-movflags", "+faststart", final)
    for old in DELIVERY.glob(final.name + ".part*"):
        old.unlink()
    h, n = hashlib.sha256(), 0
    with open(final, "rb") as src:
        while chunk := src.read(PART):
            h.update(chunk)
            (DELIVERY / f"{final.name}.part{n:02d}").write_bytes(chunk)
            n += 1
    (DELIVERY / f"{final.name}.sha256").write_text(f"{h.hexdigest()}  {final.name}\n")
    print("delivery", final, final.stat().st_size, "bytes", n, "parts", h.hexdigest())
    prev = OUT / "preview"
    prev.mkdir(exist_ok=True)
    log = prev / "x264_2pass"
    for i, (a, b) in enumerate(PREVIEW_GROUPS, 1):
        t0 = film.chapter(a).T0
        dur = film.chapter(b).T0 + film.chapter(b).duration - t0
        vb = int(min((PREVIEW_BYTES * 8 / dur - 128_000) * 0.96, 4_000_000))
        common = ["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", master, "-vf", "scale=1280:720",
                  "-c:v", "libx264", "-preset", "medium", "-b:v", vb, "-passlogfile", log]
        _ff(*common, "-pass", 1, "-an", "-f", "null", "/dev/null")
        out = prev / f"tianwen_preview_{i}_{a}-{b}.mp4"
        _ff(*common, "-pass", 2, "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out)
        print("preview", out, out.stat().st_size, "bytes", f"{dur:.1f}s")
    for f in prev.glob("x264_2pass*"):
        f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    film = load_film()
    if a.cmd == "info":
        for ch in film.chapters:
            print(f"{ch.key:10s} {ch.title:8s} start {ch.T0:7.2f}s  dur {ch.duration:6.2f}s  beats {ch.total_beats:6.1f}")
        print(f"total {film.duration:.2f}s ({film.duration / 60:.2f} min)")
    elif a.cmd == "frames":
        key, frames = a.args[0], [int(x) for x in a.args[1:]]
        for p in render_chapter(film, key, frames=frames):
            print(p)
    elif a.cmd == "render":
        keys = [c.key for c in film.chapters] if a.args[0] == "all" else a.args
        for k in keys:
            out, n = render_chapter(film, k, a.workers)
            print("rendered", out, n, "frames", flush=True)
    elif a.cmd == "music":
        from engine.music import Score
        sc = Score(film.duration)
        for ch in film.chapters:
            if ch.music:
                ch.music(sc, ch)
        OUT.mkdir(parents=True, exist_ok=True)
        sc.master(OUT / "score.wav")
        print("score.wav", round(film.duration, 2))
    elif a.cmd == "assemble":
        d = OUT / "chapters"
        lst = d / "all.txt"
        lst.write_text("".join(f"file '{c.key}.mp4'\n" for c in film.chapters))
        video = OUT / "tianwen_video.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(video)], check=True)
        master = OUT / "tianwen_master.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(OUT / "score.wav"),
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart",
                        str(master)], check=True)
        print("master", master)
    elif a.cmd == "deliver":
        deliver(film)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
