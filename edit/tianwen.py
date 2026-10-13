"""《天问》全片入口。

python3 tianwen.py frames <章节> <镜头内帧号...>   # 输出检查帧
python3 tianwen.py render <章节|all> [--workers 4]
python3 tianwen.py music                          # 生成全片配乐 out/score.wav
python3 tianwen.py assemble                       # 拼接各章 + 配乐 → out/tianwen_master.mp4
python3 tianwen.py info                           # 打印各章时长
"""
import argparse
import importlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine.film import Film, render_chapter, OUT  # noqa: E402
from engine.core import FPS  # noqa: E402

ORDER = ["prologue", "myth", "gaze", "pioneers", "race_a", "race_b", "return_", "epilogue"]


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
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
