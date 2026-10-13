"""读取 assets/ 原件：自动拼回 .partNN 分片并校验 sha256，缓存到 edit/.cache/。"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CACHE = ROOT / "edit" / ".cache"
Image.MAX_IMAGE_PIXELS = None


def resolve(rel):
    """返回原件路径；若原件以分片入库，拼回到缓存目录并校验。"""
    p = ASSETS / rel
    if p.exists():
        return p
    parts = sorted(p.parent.glob(p.name + ".part[0-9][0-9]"))
    if not parts:
        raise FileNotFoundError(rel)
    out = CACHE / rel
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out, "wb") as f:
        for part in parts:
            data = part.read_bytes()
            h.update(data)
            f.write(data)
    sha_file = p.parent / (p.name + ".sha256")
    if sha_file.exists():
        expect = sha_file.read_text().split()[0]
        if h.hexdigest() != expect:
            out.unlink()
            raise ValueError(f"sha256 mismatch: {rel}")
    return out


def find(asset_id, candidate="01"):
    """按编号找原件，如 find('PD-11') → assets/PD/PD-11/PD-11_01_….tif（或分片）。"""
    folder = ASSETS / asset_id[:2] / asset_id
    for f in sorted(folder.iterdir()):
        name = f.name
        if name.startswith(f"{asset_id}_{candidate}_"):
            for suffix in (".sha256",) + tuple(f".part{i:02d}" for i in range(100)):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            return f"{asset_id[:2]}/{asset_id}/{name}"
    raise FileNotFoundError(asset_id)


def load_image(asset_id, candidate="01", max_edge=3000):
    """读取图片原件为 float32 RGB [0,1]，长边缩到 max_edge 以内。"""
    im = Image.open(resolve(find(asset_id, candidate)))
    if im.mode in ("I;16", "I;16B", "I"):
        im = im.point(lambda v: v / 256).convert("L")
    im = im.convert("RGB")
    s = max_edge / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0
