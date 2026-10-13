"""生成《天问》全片拍点表：edit/beatmap.json 和 docs/BEAT_MAP.md。

每个段落给出 BPM（可渐变）和小节数，按 4/4 拍展开成逐拍的时间和帧号（30fps）。
剪辑时所有切点、冲击帧、字幕入场都从这张表取。
"""
import json
from pathlib import Path

FPS = 30
ROOT = Path(__file__).resolve().parent.parent

# (段落, 起始 BPM, 结束 BPM, 小节数, 渐变占用的小节数, 说明)
SECTIONS = [
    ("序章", 60, 60, 6, 0, "心跳脉冲；第 4 小节强拍点火（H1）；第 5 小节倒带"),
    ("壹 神话", 72, 72, 18, 0, "章节卡 2 小节 + 四个神话段落各 4 小节"),
    ("贰 仰望", 84, 84, 18, 0, "章节卡 2 小节 + 四位人物各 4 小节"),
    ("叁 先驱", 96, 112, 44, 24, "前 24 小节从 96 渐快到 112，之后保持；最后一小节强拍音爆（H3），接一拍静音"),
    ("肆 4A 第一连发", 120, 120, 16, 0, "每拍一切"),
    ("肆 4B 登月", 120, 120, 20, 0, "第 9 小节强拍阿波罗 11 号升空（H4）"),
    ("肆 4C 同一个月", 120, 120, 16, 0, "鼓组撤掉，半速感"),
    ("肆 4D 远方", 120, 120, 24, 0, "第 17—19 小节为暗淡蓝点静音（H6），约 6 秒"),
    ("肆 4E 长住太空", 128, 128, 14, 0, "哥伦比亚号静默后神舟五号起势"),
    ("肆 4F 加速", 128, 128, 20, 0, "每 2 拍一切，最后 4 个镜头每拍一切"),
    ("伍 回来，再出发", 100, 100, 38, 0, "半速重拍；英雄时刻落在小节第一拍（H8、H9、H10）"),
    ("尾声", 60, 60, 10, 0, "自由速度，按 60 BPM 估算；第 1 小节强拍 47 → 33（H11）"),
]


def bpm_at(bar_pos, start, end, ramp_bars):
    """段落内第 bar_pos 小节（可为小数）处的瞬时 BPM。"""
    if ramp_bars <= 0 or start == end:
        return start
    k = min(bar_pos / ramp_bars, 1.0)
    return start + (end - start) * k


def build():
    beats, sections, t = [], [], 0.0
    for name, b0, b1, bars, ramp, note in SECTIONS:
        start_t = t
        for bar in range(bars):
            for beat in range(4):
                pos = bar + beat / 4
                bpm = bpm_at(pos, b0, b1, ramp)
                beats.append({
                    "section": name,
                    "bar": bar + 1,
                    "beat": beat + 1,
                    "time_s": round(t, 4),
                    "frame": round(t * FPS),
                    "bpm": round(bpm, 2),
                    "downbeat": beat == 0,
                })
                t += 60.0 / bpm
        sections.append({
            "section": name, "bpm": f"{b0}" if b0 == b1 else f"{b0}→{b1}",
            "bars": bars, "start_s": round(start_t, 3), "end_s": round(t, 3),
            "start_frame": round(start_t * FPS), "end_frame": round(t * FPS),
            "note": note,
        })
    return beats, sections, t


def fmt(sec):
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def main():
    beats, sections, total = build()
    (ROOT / "edit" / "beatmap.json").write_text(
        json.dumps({"fps": FPS, "total_s": round(total, 3), "sections": sections, "beats": beats},
                   ensure_ascii=False, indent=1))
    lines = [
        "# 《天问》全片拍点表",
        "",
        f"由 `edit/beatmap.py` 生成。4/4 拍，{FPS}fps，全片约 {fmt(total)}。"
        "逐拍的时间和帧号见 `edit/beatmap.json`；修改段落的 BPM 或小节数后重新运行脚本即可。",
        "",
        "| 段落 | BPM | 小节 | 起点 | 终点 | 起始帧 | 说明 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in sections:
        lines.append(f"| {s['section']} | {s['bpm']} | {s['bars']} | {fmt(s['start_s'])} | "
                     f"{fmt(s['end_s'])} | {s['start_frame']} | {s['note']} |")
    (ROOT / "docs" / "BEAT_MAP.md").write_text("\n".join(lines) + "\n")
    print(f"total {fmt(total)}, {len(beats)} beats")


if __name__ == "__main__":
    main()
