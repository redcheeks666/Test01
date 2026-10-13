# 《天问》成片

| 文件 | 说明 |
|---|---|
| `tianwen_1080p.mp4.partNN` | 上传版成片：1920×1080，30fps，H.264 High，AAC 320k 48kHz；按 90 MiB 二进制分片 |
| `tianwen_1080p.mp4.sha256` | 拼回后整片的校验值 |
| `tianwen_preview_720p.mp4` | 整片预览：1280×720，单文件不分片，下载即可直接播放 |

在本目录拼回并校验：

```sh
cat tianwen_1080p.mp4.part[0-9][0-9] > tianwen_1080p.mp4
shasum -a 256 -c tianwen_1080p.mp4.sha256
```

由 `python3 edit/tianwen.py deliver` 从 `edit/out/tianwen_master.mp4` 生成。发布时的署名文案见 `docs/CREDITS_post.md`。
