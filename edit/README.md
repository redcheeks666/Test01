# 剪辑工程

## 全片（`tianwen.py`）

```bash
cd edit
python3 tianwen.py info                    # 各章时长
python3 tianwen.py frames <章节> 100 200    # 输出检查帧到 out/chapters/
python3 tianwen.py render all --workers 4  # 逐章并行渲染 → out/chapters/<章节>.mp4
python3 tianwen.py music                   # 全片配乐与音效 → out/score.wav
python3 tianwen.py assemble                # 拼接 + 混音 → out/tianwen_master.mp4
python3 tianwen.py deliver                 # 上传版 → ../delivery/（90 MiB 分片 + sha256）；720p 分段预览 → out/preview/
```

| 目录 | 说明 |
| --- | --- |
| `engine/core.py` | 画面工具、文字与三级字幕（浅色字自动按背景亮度加暗底）、统一章节卡、左下角年份标尺、后期 |
| `engine/media.py` | 原件读取、视频片段抽帧缓存、变速曲线 |
| `engine/shots.py` | 通用镜头：照片运镜、视频、分屏、纪念镜头、甩镜 |
| `engine/film.py` | 全片时间轴（每章自己的 BPM，可渐变）与多进程渲染 |
| `engine/music.py` | 合成乐器库、真实音效、BGM 自动电平（0 到 −6 dBFS）、前瞻真峰值限幅（4 倍过采样，−1.5 dBTP，无削波） |
| `chapters/*.py` | 各章镜头清单与配乐提示：序章、神话、仰望、先驱、竞赛（前后两段）、回来再出发、尾声 |


| 文件 | 说明 |
| --- | --- |
| `beatmap.py` | 生成全片拍点表 `beatmap.json` 和 `docs/BEAT_MAP.md`。改了段落的 BPM 或小节数就重新运行 |
| `fonts/` | 字体：马善政毛笔楷书（大字关键词）、思源宋体（解释与小字）、IBM Plex Mono（日期戳）、志莽行书（备用）。均为 SIL OFL 开源许可 |
| `assets.py` | 读取 `assets/` 原件，自动拼回分片并校验 sha256 |
| `ch3_demo/` | 先驱章节奏样片 |

## 先驱章节奏样片（`ch3_demo/`）

用来验证 `docs/EDIT_RHYTHM.md` 里的手法：卡点、变速、照片运镜、冲击帧、三级文字节奏。约 37 秒，96 BPM 渐快到 112 BPM。BGM 自动电平在 0 到 −6 dBFS 之间。

```bash
cd edit/ch3_demo
python3 music.py    # 合成配乐和音效 → out/music.wav
python3 render.py   # 逐帧渲染并合成 → out/ch3_demo.mp4
python3 render.py 120 300   # 只渲染第 120、300 帧，输出 jpg 做检查
```

依赖：Python 3、numpy、scipy、opencv-python-headless、Pillow、ffmpeg。

| 拍 | 镜头 | 手法 |
| --- | --- | --- |
| 0—8 | 3-01 章节卡 | "先驱"毛笔大字浮现 → 定音鼓滚奏渐强 → 盖"天问"印 → 最后半拍冲进墨迹 |
| 8—20 | 3-02 万户 | 月亮前的剪影插画慢推，引信燃烧 → 第 15.5 拍抽空 → 第 16 拍点火，椅下火箭向下喷焰 → 第 17 拍离地，飞出画面 → 尾烟盖满画面转场（此段多一小节，后面的拍号顺延 4 拍） |
| 16—24 | 3-03 热气球 | 第 16 拍爆炸白烟铺满 → 从气球局部拉出整幅版画（揭示） |
| 24—28 | 3-06 李林塔尔 | 甩镜入画 + 向右上漂移 |
| 28—36 | 3-07 1903 分屏 | 左边莱特兄弟每 2 拍切一个局部（细节拆解）；右边火箭方程逐笔写出 |
| 36—40 | 3-08 纽约时报 | 报纸旋转冲入；第 38 拍急推到社论标题 |
| 40—44 | 3-09 戈达德 | 从人物向上摇到火箭 |
| 44—48 | 3-11 V-2 拍下的地球 | 从黑场和噪点中浮现 |
| 48—52 | 3-12 X-1 | 加速推进 + 速度线 → 第 50 拍音爆（白闪、震动、色散）切回全景 |
| 52— | 黑场 → 竞赛 | 一拍完全静音 → "竞赛"章节卡随定音鼓和铜管重音落下 |

**这一版的限制：**

- 照片已换成 `assets/` 里的原件（通过 `edit/assets.py` 读取，分片文件自动拼回并校验）。
- 配乐是代码合成的；音效优先用 `assets/FX` 的真实录音，缺的才用合成。
- 凯利、凡尔纳、皮卡尔三个镜头缺素材，这一版先跳过。
- 《纽约时报》版面是重绘的示意，不是原版扫描。
