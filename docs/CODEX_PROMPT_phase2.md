# 任务：《天问》素材入库 + 补充收集（阶段二）

先读仓库里的 `assets_review/REVIEW_claude.md`（Claude 对阶段一的审核意见）。本阶段做两件事：**A. 把原素材无损推到仓库**；**B. 按审核意见补充收集**。B 下载的新文件同样按 A 的规则入库。

仓库：`https://github.com/redcheeks666/Test01`，分支：`claude/hello-nv3sez`。开始前先 `git pull --rebase origin claude/hello-nv3sez`。

---

## A. 原素材无损入库

### 规则

1. 在仓库根目录建 `assets/`，目录结构和你本地的 `/Users/mashazhenta/CodeXProject/Vibe知识大赏/assets/` 完全一致（`SX/ NA/ SU/ CN/ IN/ PD/ FX/ AU/<编号>/<文件>`）。只复制原素材文件，不复制 previews、manifest 这些（它们在 `assets_review/` 里）。
2. **不超过 95 MB 的文件**：原样复制，直接提交。
3. **超过 95 MB 的文件**：不提交原文件，改为提交二进制分片：
   ```bash
   split -b 90m -d -a 2 <文件> <文件>.part
   shasum -a 256 <文件> > <文件>.sha256
   ```
   得到 `<文件>.part00`、`<文件>.part01`……和 `<文件>.sha256`。把原文件路径追加到仓库根目录的 `.gitignore`，确保它不会被 `git add`。
4. 在 `assets/SPLIT_FILES.md` 里列出所有被分片的文件：原路径、原大小、分片数、sha256。
5. **X 直播回放和其他超过 10 分钟的长视频**：不整段入库。整段回放放在本地的 `assets/<类别>/<编号>/_full_broadcast/` 里（这个目录已在 `.gitignore` 中）。按关键画面用 ffmpeg 无损截取片段：
   ```bash
   ffmpeg -ss <入点> -to <出点> -i <整段文件> -c copy -map 0 <片段文件>
   ```
   - 每个关键画面前后各留 10 秒余量；
   - 片段命名：`<编号>_<候选序号>_<简述>_cut<两位序号>_<来源>.<扩展名>`；
   - 在 manifest 里给每个片段单独占一行，填 `excerpt_of`（整段文件路径）、`excerpt_in`、`excerpt_out`（秒）；
   - 片段超过 95 MB 时，按第 3 条分片。
6. 不转码、不重新压缩。`-c copy` 截取后如果片段开头画面花屏（关键帧问题），把入点往前挪到最近的关键帧，不要改成重新编码。

### SpaceX 第 14 次试飞直播（SX-01、SX-03）的截取点

REPORT.md 里给的是任务时码（T±）。先在整段回放里找到 T−0 对应的播放器时码，然后截这些片段：

| 片段 | 任务时码 | 内容 |
| --- | --- | --- |
| SX-01 cut01 | T−00:01:00 至 T+00:03:00 | 塔旁全景、点火、离塔、上升，包括所有机位切换 |
| SX-03 cut01 | T+00:07:00 至 T+00:10:00 | 入轨后舱外摄像头看到的地球边缘 |
| SX-03 cut02 | T+00:34:00 至 T+00:40:00 | 星链卫星开始部署 |
| SX-03 cut03 | 你在回放里看到的其他"地球 + 星舰"的好画面 | 每段 30–90 秒，最多 3 段 |

每个片段的 key_moments 写清楚：点火、离塔、各次机位切换、直播叠加层出现和消失的时码。

### 推送

- 按类别分批提交和推送，**每次推送不超过 1 GB**。建议顺序：AU+FX → PD → CN+IN+SU → NA（分两批）→ SX。
- 提交信息写清楚，如"Add original assets: NA (part 1/2)"。
- 每次推送前用 `git status` 和 `git diff --cached --stat` 确认：没有超过 95 MB 的单个文件，没有 `_full_broadcast/` 里的文件。
- 全部推完后，重新生成 `assets_review/manifest.csv`、`inventory.csv`、`INVENTORY.md`（加一列 `in_repo`：直接入库 / 分片入库 / 只在本地），并在 `INVENTORY.md` 的总览里写出仓库中原素材的总大小。

---

## B. 补充收集

按 `REVIEW_claude.md` 的"必须补的"一节执行，优先级从高到低。所有来源、许可、命名、预览图、manifest 的规则沿用 `docs/CODEX_PROMPT_v2.md`。和 v2 不同的地方：

1. **SX-02、SX-06**：SpaceX 官方 Flickr 没有的，可以从 **SpaceX 官方 X 账号的帖子里下载原图**（用户已同意获取 SpaceX 官方 X 内容）。记录帖子链接。
2. **SX-07**：替换现有候选。要火箭占画面主体的 Demo-2 发射照片（NASA 或 SpaceX Flickr）或视频。
3. **CN-07**：补 2026-07-10 长征十号乙海上网系回收的画面。官网图太小的话，可以下载国家航天局、中国航天科技集团官方账号（微博、微信视频号、B 站、抖音）发布的视频，在 manifest 的 notes 里记录账号和链接，在 REPORT 里列入"待用户确认"。
4. **PD-04**：Gallica 返回 403 时，可以用 Wikimedia Commons 上 Hetzel 版原书插图的扫描（1865/1872 年出版，公有领域）。只要原书扫描，不要重绘。
5. **SU-01—SU-07**：优先试 Wikimedia Commons 上的俄新社档案（RIA Novosti archive，CC BY-SA 3.0），文件页上必须有这个许可模板。找不到也没关系，Claude 会用代码绘制。
6. **NA-03**：可以接受 NASA SVS 的上弦月帧（LRO 数据生成），description 写"SVS 可视化（LRO 数据）"。
7. **音效**：FX-12（电传打字机）、FX-13（印刷机）、FX-15（音爆）、FX-17（VHS 故障音）优先，每个至少 3 个候选；FX-01 补回车键单击。
8. **CN-01、CN-03、CN-06**：继续找同一画面的高清版本（中国探月与深空探测网、新华社、人民网的原图）。找不到就保留现有文件。
9. **NA-31_02**：找更高清的 Artemis I 发射照片。

## REPORT.md

在现有 REPORT.md 末尾加一节"阶段二"：

1. 入库情况：直接入库多少个、分片入库多少个、只在本地多少个，仓库里原素材的总大小。
2. 补充收集的结果：逐个编号写新增了什么，还缺什么。
3. 新的"待用户确认"事项。

## 完成标准

- `assets/` 已全部推送，仓库里没有超过 95 MB 的单个文件。
- `assets/SPLIT_FILES.md` 列出所有分片文件和 sha256。
- X 直播只有截取的片段入库，manifest 填了 `excerpt_of`、`excerpt_in`、`excerpt_out` 和 key_moments。
- manifest、inventory、INVENTORY.md、REPORT.md 已更新，previews 里每个新文件都有预览图。
