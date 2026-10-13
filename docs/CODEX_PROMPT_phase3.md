# 任务：《天问》替换素材下载（阶段三）

目的：片中"天宫""嫦娥六号"两个镜头原先用的是中国载人航天工程办公室、新华社的图，前者明确要求授权且禁止二次加工，后者需新华社许可。改用许可清楚的同题材素材。

仓库：`https://github.com/redcheeks666/Test01`，分支：`claude/hello-nv3sez`。开始前先 `git pull --rebase origin claude/hello-nv3sez`。

入库规则、命名、manifest 字段沿用 `docs/CODEX_PROMPT_phase2.md`：原件不转码，单文件超过 95 MB 按 90 MiB 分片并附 sha256，每个文件在 `assets_review/manifest.csv` 占一行，`license`、`credit_line`、`source_page_url`、`direct_file_url` 必填。

## 必须下载

| 编号 | 内容 | 来源 | 署名 |
| --- | --- | --- | --- |
| NA-32 | NASA 月球勘测轨道飞行器（LRO）2024-06-07 拍到的月背嫦娥六号着陆器（Apollo 盆地南缘，着陆器周围有发动机吹出的亮区） | NASA Goddard 文章 "NASA's LRO Spots China's Chang'e 6 Spacecraft on Lunar Far Side"（nasa.gov），以及 LROC 团队网站上的同一张图 | NASA/GSFC/Arizona State University |

要求：
- 下载**最高分辨率**的版本。优先 LROC 网站提供的全分辨率裁切图；如果有更大范围的原始 NAC 图像（能看到着陆点周围的环形山），一并下载，作为候选 02。
- 在 key_moments 里写出着陆器在图中的像素坐标，方便推近。

## 天宫：先找，找不到就如实报告

只收**明确标注开放许可**的真实照片：公有领域、CC0、CC BY；CC BY-SA 也可以，但要在 notes 里标明"署名-相同方式共享"。

可以找的方向：
- Wikimedia Commons 上的天宫照片，逐个打开文件页核对许可框和作者。注意区分真实照片和三维渲染图（例如作者 Shujianyang 的天宫图很可能是渲染图，如果收了，在 description 里写明"渲染图"）。
- 中国新闻社等机构以 CC BY 方式发布的视频或截图（必须在原发布页或 Commons 文件页看到许可声明）。
- 天文爱好者用望远镜拍到的天宫过境照片，只收许可页明确写了开放许可的。

**不要**收新闻机构、载人航天办公室官网、社交媒体转载的图。找不到符合条件的，就在 REPORT 里写"天宫：未找到开放许可的真实照片"，并列出查过的地方。用户这边的备选方案是由 Claude 代码绘制。

## 可选（时间允许再做）

同样换成 NASA 轨道器视角，让中国探测器这几张图的来源统一：

| 编号 | 内容 | 署名 |
| --- | --- | --- |
| NA-33 | LRO 拍到的冯·卡门撞击坑里的嫦娥四号着陆器与玉兔二号（2019） | NASA/GSFC/Arizona State University |
| NA-34 | 火星勘测轨道飞行器 HiRISE 拍到的祝融号着陆平台与火星车（2021-06-06 或 2021-06-11） | NASA/JPL-Caltech/University of Arizona |

## 完成后

- 推送到同一分支，提交信息如 "Add replacement assets: NA-32 (LRO Chang'e 6)"。
- 在 `assets_review/REPORT.md` 末尾加"阶段三"一节：下载了什么、每个文件的许可依据（贴许可原文所在链接）、天宫的查找结果。
