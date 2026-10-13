"""扫描剪辑代码里实际引用的素材，对照 assets_review/manifest.csv 生成 docs/ASSET_SOURCES.md。

python3 asset_table.py
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CH_ORDER = ["prologue", "myth", "gaze", "pioneers", "race_a", "race_b", "return_", "epilogue"]
CH_NAME = {"prologue": "序章", "myth": "神话", "gaze": "仰望", "pioneers": "先驱", "race_a": "竞赛（上）",
           "race_b": "竞赛（下）", "return_": "回来，再出发", "epilogue": "尾声"}
# 从大图裁出的工作区缓存 → 对应原件
CACHE_SRC = {"na32_change6_work": ("NA-32", "02"), "na35_zhurong_work": ("NA-35", "01"),
             "na36_change4_work": ("NA-36", "02")}

PAT_CALL = re.compile(r'\b(?:image|img)\(\s*"([A-Z]{2}-\d+)"(?:\s*,\s*"(\d+)")?')
PAT_PATH = re.compile(r'"(?:[A-Z]{2})/([A-Z]{2}-\d+)/\1_(\d+)_[^"]+"')
PAT_FIND = re.compile(r'find\(\s*"([A-Z]{2}-\d+)"\s*,\s*"(\d+)"')
PAT_CACHE = re.compile(r'"(na\d+_[a-z0-9_]+?)(?:\.png)?"')


def refs(src):
    out = set()
    for m in PAT_CALL.finditer(src):
        out.add((m.group(1), m.group(2) or "01"))
    for pat in (PAT_PATH, PAT_FIND):
        for m in pat.finditer(src):
            out.add((m.group(1), m.group(2)))
    for m in PAT_CACHE.finditer(src):
        if m.group(1) in CACHE_SRC:
            out.add(CACHE_SRC[m.group(1)])
    return out


def main():
    with open(ROOT / "assets_review" / "manifest.csv", newline="", encoding="utf-8") as f:
        man = {(r["asset_id"], r["candidate"]): r for r in csv.DictReader(f)}
    used = defaultdict(list)
    for key in CH_ORDER:
        for ref in refs((HERE / "chapters" / f"{key}.py").read_text(encoding="utf-8")):
            used[ref].append(CH_NAME[key])
    sample = any("sample_v0" in (HERE / "chapters" / f"{k}.py").read_text(encoding="utf-8") for k in CH_ORDER)

    groups = {"SX": "SpaceX", "NA": "NASA", "AU": "真实录音", "FX": "音效", "PD": "古籍与档案（公有领域）",
              "CN": "中国机构", "WM": "Wikimedia Commons 个人作品"}
    lines = ["# 片中使用的素材来源", "",
             "由 `edit/asset_table.py` 扫描剪辑代码生成：只列片中实际用到的素材。来源、许可、署名取自 "
             "`assets_review/manifest.csv`。发帖署名文案见 `docs/CREDITS_post.md`。", ""]
    for g, title in groups.items():
        rows = sorted((k for k in used if k[0].startswith(g)), key=lambda k: (int(k[0][3:]), k[1]))
        if not rows:
            continue
        lines += [f"## {title}（{len(rows)}）", "", "| 编号 | 内容 | 来源 | 许可 | 署名 | 用在 |", "|---|---|---|---|---|---|"]
        for k in rows:
            r = man.get(k, {})
            cell = lambda s: (s or "—").replace("|", "/").replace("\n", " ")
            lines.append(f"| {k[0]}/{k[1]} | {cell(r.get('description'))} | {cell(r.get('source_org'))} | "
                         f"{cell(r.get('license'))} | {cell(r.get('credit_line'))} | {'、'.join(dict.fromkeys(used[k]))} |")
        lines.append("")
    lines += ["## 其他", "", "| 内容 | 来源 |", "|---|---|",
              "| 配乐 | 程序合成（`edit/engine/music.py`） |",
              "| 万户、苏联航天器与航天员、神舟五号、月船 3 号、凯利滑翔机、皮卡尔气球、浑天仪等画面 | 代码绘制 |",
              "| 字体 | 马善政楷书、思源宋体、IBM Plex Mono（均为 SIL OFL 开源许可） |"]
    if sample:
        lines.append("| 神话段落部分画面 | 用户样片 `reference/sample_v0.mp4` |")
    out = ROOT / "docs" / "ASSET_SOURCES.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out, sum(1 for _ in used), "assets")


if __name__ == "__main__":
    main()
