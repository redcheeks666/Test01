"""《天问》全片入口。

python3 tianwen.py frames <章节> <镜头内帧号...>   # 输出检查帧
python3 tianwen.py render <章节|all> [--workers 4]
python3 tianwen.py music                          # 生成全片配乐 out/score.wav
python3 tianwen.py assemble                       # 拼接各章 + 配乐 → out/tianwen_master.mp4
python3 tianwen.py info                           # 打印各章时长
python3 tianwen.py deliver [hd|preview]           # 上传版 1080p（90 MiB 分片 + sha256）与整片 720p 预览（单文件 <95MB）
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
PREVIEW_BYTES = 88_000_000                   # 整片预览单文件，不分片也低于 95 MB


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


HD_V = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", "12M", "-bufsize", "24M",
        "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p", "-g", "60"]


def _encode_chapters(film, tag, vargs, two_pass=False):
    """逐章编码视频（不带音轨）。章节原片没变、参数没变就沿用上次的结果，所以改一章只重编这一章。"""
    d = OUT / f"enc_{tag}"
    d.mkdir(exist_ok=True)
    sig = " ".join(map(str, vargs))
    files = []
    for ch in film.chapters:
        src, dst, stamp = OUT / "chapters" / f"{ch.key}.mp4", d / f"{ch.key}.mp4", d / f"{ch.key}.args"
        fresh = dst.exists() and stamp.exists() and stamp.read_text() == sig and dst.stat().st_mtime > src.stat().st_mtime
        if not fresh:
            if two_pass:
                log = d / f"{ch.key}_2pass"
                _ff("-i", src, *vargs, "-passlogfile", log, "-pass", 1, "-an", "-f", "null", "/dev/null")
                _ff("-i", src, *vargs, "-passlogfile", log, "-pass", 2, "-an", dst)
                for f in d.glob(f"{ch.key}_2pass*"):
                    f.unlink()
            else:
                _ff("-i", src, *vargs, "-an", dst)
            stamp.write_text(sig)
            print("encoded", tag, ch.key, flush=True)
        files.append(dst)
    return files


def _concat_with_score(files, audio_args, out):
    """各章视频直接拼接（不重编），配乐整条编码一次再封装；音轨补静音到与画面等长，不丢最后一帧。"""
    lst = OUT / f"{out.stem}_concat.txt"
    lst.write_text("".join(f"file '{f}'\n" for f in files))
    _ff("-f", "concat", "-safe", "0", "-i", lst, "-i", OUT / "score.wav", "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", *audio_args, "-af", "apad", "-shortest", "-movflags", "+faststart", out)
    lst.unlink()


def deliver(film, what=("hd", "preview")):
    """上传版：1080p H.264 High + AAC 320k，按 90 MiB 分片并附 sha256 入库。
    预览：720p 两遍编码，整片单文件低于 95 MB，直接入库。两者都按章节编码后拼接。"""
    DELIVERY.mkdir(exist_ok=True)
    if "hd" in what:
        final = DELIVERY / "tianwen_1080p.mp4"
        _concat_with_score(_encode_chapters(film, "hd", HD_V), ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"], final)
        split_file(final)
    if "preview" in what:
        out = DELIVERY / "tianwen_preview_720p.mp4"
        vb = int((PREVIEW_BYTES * 8 / film.duration - 128_000) * 0.96)
        pv = ["-vf", "scale=1280:720", "-c:v", "libx264", "-preset", "medium", "-b:v", vb, "-pix_fmt", "yuv420p"]
        _concat_with_score(_encode_chapters(film, "preview", pv, two_pass=True), ["-c:a", "aac", "-b:a", "128k"], out)
        print("preview", out, out.stat().st_size, "bytes")


def split_file(final):
    """按 90 MiB 二进制分片，附整片 sha256（shasum -c 格式）。"""
    for old in final.parent.glob(final.name + ".part*"):
        old.unlink()
    h, n = hashlib.sha256(), 0
    with open(final, "rb") as src:
        while chunk := src.read(PART):
            h.update(chunk)
            (final.parent / f"{final.name}.part{n:02d}").write_bytes(chunk)
            n += 1
    (final.parent / f"{final.name}.sha256").write_text(f"{h.hexdigest()}  {final.name}\n")
    print("delivery", final, final.stat().st_size, "bytes", n, "parts", h.hexdigest())


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
        deliver(film, tuple(a.args) or ("hd", "preview"))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
