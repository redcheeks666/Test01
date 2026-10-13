"""全片配乐与音效：合成乐器库、真实音效读取、两条总线（BGM / 音效）、BGM 自动电平、前瞻限幅。

所有乐器与音效函数返回单声道 float64 数组；Score.add 负责定位、增益、声像。
"""
import subprocess
from functools import lru_cache

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import butter, fftconvolve, sosfilt

from .core import ROOT

SR = 48000
ASSETS = ROOT / "assets"
rng = np.random.default_rng(7)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def filt(x, kind, f, order=4):
    return sosfilt(butter(order, f, btype=kind, fs=SR, output="sos"), x)


def noise(dur):
    return rng.standard_normal(max(int(dur * SR), 1))


def tt(dur):
    return np.arange(max(int(dur * SR), 1)) / SR


def env(dur, a, d, s_level=0.0, r=0.05):
    x = tt(dur)
    e = np.where(x < a, x / max(a, 1e-4), s_level + (1 - s_level) * np.exp(-(x - a) / max(d, 1e-4)))
    rn = int(r * SR)
    if rn and len(e) > rn:
        e[-rn:] *= np.linspace(1, 0, rn)
    return e


def saw(f, dur, n_harm=14, detune_cents=(0, -7, 7)):
    x = tt(dur)
    out = np.zeros_like(x)
    for c in detune_cents:
        ff = f * 2 ** (c / 1200)
        for k in range(1, n_harm + 1):
            if ff * k > SR / 2.2:
                break
            out += np.sin(2 * np.pi * ff * k * x + rng.uniform(0, 6.28)) / k
    return out / len(detune_cents)


# ---------------- 乐器 ----------------

def strings_short(m, dur):
    s = filt(saw(midi(m), dur + 0.15), "low", 1800)
    return s * env(dur + 0.15, 0.012, 0.16, 0.0, 0.08)


def strings_long(m, dur, bright=1800):
    s = filt(saw(midi(m), dur + 0.3, 12, (0, -6, 6)), "low", bright)
    x = tt(dur + 0.3)
    s *= 1 + 0.004 * np.sin(2 * np.pi * 5.2 * x)
    return s * env(dur + 0.3, 0.25, 99, 1.0, 0.35)


def pad(ms, dur, cutoff=1400, attack=0.5):
    s = sum(saw(midi(m), dur, 10, (0, -9, 9)) for m in ms)
    return filt(s, "low", cutoff) * env(dur, attack, 99.0, 1.0, min(0.6, dur / 3)) / len(ms)


def choir(ms, dur, attack=0.8):
    """合唱铺底：锯齿波经元音"啊"的共振峰滤波。"""
    out = np.zeros(len(tt(dur)))
    for m in ms:
        s = saw(midi(m), dur, 20, (0, -11, 11, -4, 4))
        x = tt(dur)
        s *= 1 + 0.01 * np.sin(2 * np.pi * 5 * x + rng.uniform(0, 6))
        v = filt(s, "band", [600, 820], 2) * 1.0 + filt(s, "band", [1100, 1350], 2) * 0.6 + filt(s, "band", [2450, 2750], 2) * 0.3
        out += v
    return out * env(dur, attack, 99, 1.0, min(0.8, dur / 3)) / len(ms) * 3


def horn(m, dur):
    x = tt(dur + 0.3)
    f = midi(m) * (1 + 0.004 * np.sin(2 * np.pi * 5 * x) * np.clip(x / 0.4, 0, 1))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = sum(np.sin(k * ph) * a for k, a in [(1, 1), (2, .5), (3, .32), (4, .18), (5, .1), (6, .05)])
    return filt(s, "low", 2200) * env(dur + 0.3, 0.08, 3.0, 0.85, 0.3)


def brass_stab(ms, dur=0.5):
    s = sum(saw(midi(m), dur, 18, (0, -5, 5)) for m in ms)
    return filt(s, "low", 3200) * env(dur, 0.006, 0.18, 0.0, 0.1) / len(ms)


def pluck(m, dur=2.5, bright=1.0, decay=1.0, slide=0.0):
    """拨弦（古琴、拨奏）：泛音逐级衰减，可带起音滑音。"""
    x = tt(dur)
    f0 = midi(m) * (1 + slide * np.exp(-x * 12))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    s = np.zeros_like(x)
    for k in range(1, 13):
        s += np.sin(k * ph * (1 + 0.0007 * k * k)) / k ** (1.3 - 0.3 * bright) * np.exp(-x * (0.9 + 0.55 * k) / decay)
    att = filt(noise(0.03), "band", [800, 4000]) * np.exp(-tt(0.03) * 120) * 0.3
    s[: len(att)] += att
    return filt(s, "low", 3500 * bright) * 0.5


def xiao(m, dur, vib=1.0):
    """箫：正弦 + 少量泛音 + 气声，0.3 秒后起颤音。"""
    x = tt(dur + 0.25)
    vibr = 1 + 0.006 * vib * np.sin(2 * np.pi * 5.2 * x) * np.clip((x - 0.3) / 0.4, 0, 1)
    ph = 2 * np.pi * np.cumsum(midi(m) * vibr) / SR
    s = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    breath = filt(noise(dur + 0.25), "band", [midi(m) * 0.8, midi(m) * 3.5]) * 0.12
    return (s + breath) * env(dur + 0.25, 0.12, 99, 1.0, 0.25) * 0.6


def gong(dur=6.0, f=92):
    x = tt(dur)
    s = np.zeros_like(x)
    for r, a, d in [(1, 1, 0.5), (1.48, 0.7, 0.7), (1.95, 0.5, 0.9), (2.6, 0.35, 1.2), (3.4, 0.25, 1.6), (4.2, 0.18, 2.0)]:
        fb = f * r * (1 + 0.012 * (1 - np.exp(-x * 1.5)))
        s += a * np.sin(2 * np.pi * np.cumsum(fb) / SR) * np.exp(-x * d)
    hit = filt(noise(dur), "low", 1200) * np.exp(-x * 25) * 0.6
    return (s + hit) * 0.5


def bell(m, dur=3.0):
    x = tt(dur)
    f = midi(m)
    s = sum(a * np.sin(2 * np.pi * f * r * x) * np.exp(-x * d) for r, a, d in
            [(1, 1, 1.2), (2.76, 0.4, 2.5), (5.4, 0.2, 4.0), (8.9, 0.1, 6.0)])
    return s * 0.4


def timpani(m, dur=1.6, vel=1.0):
    x = tt(dur)
    f = midi(m) * (1 + 0.06 * np.exp(-x * 25))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-x * 2.2) + 0.5 * np.sin(1.5 * ph) * np.exp(-x * 5) + 0.3 * np.sin(1.99 * ph) * np.exp(-x * 7)
    hit = filt(noise(dur), "low", 900) * np.exp(-x * 60) * 0.6
    return (s + hit) * vel


def kick(vel=1.0):
    d = 0.5
    x = tt(d)
    f = 45 + 80 * np.exp(-x * 30)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 6)
    click = filt(noise(d), "band", [1500, 5000]) * np.exp(-x * 200) * 0.3
    return (s + click) * vel


def snare(vel=1.0):
    d = 0.18
    x = tt(d)
    n = filt(noise(d), "band", [1200, 6000]) * np.exp(-x * 28)
    body = np.sin(2 * np.pi * 190 * x) * np.exp(-x * 40) * 0.5
    return (n + body) * vel


def hat(vel=1.0, open_=False):
    d = 0.25 if open_ else 0.05
    x = tt(d)
    return filt(noise(d), "high", 7000) * np.exp(-x * (12 if open_ else 80)) * vel * 0.6


def taiko(vel=1.0, f=62):
    d = 1.2
    x = tt(d)
    fr = f * (1 + 0.4 * np.exp(-x * 18))
    s = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-x * 3.5)
    skin = filt(noise(d), "low", 600) * np.exp(-x * 30) * 0.8
    return (s + skin) * vel


def cymbal(dur=2.5):
    x = tt(dur)
    return filt(noise(dur), "high", 5000) * np.exp(-x * 1.8)


# ---------------- 音效 ----------------

def stamp():
    x = tt(0.35)
    return np.sin(2 * np.pi * 110 * x) * np.exp(-x * 22) + filt(noise(0.35), "low", 1500) * np.exp(-x * 40) * 0.6


def boom(dur=3.0, f0=55, f1=26, crack=0.4):
    """低频冲击。crack 控制高频爆裂成分（已压低，避免刺耳）。"""
    x = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-x * 2.5)
    low = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 1.1)
    rumble = filt(noise(dur), "low", 220) * np.exp(-x * 1.4) * 1.2
    cr = filt(filt(noise(dur), "high", 900), "low", 4000) * np.exp(-x * 30) * crack
    return low * 1.0 + rumble + cr


def whoosh(dur=0.7, f_lo=300, f_hi=3000):
    x = tt(dur)
    shape = np.sin(np.pi * np.clip(x / dur, 0, 1)) ** 2
    return filt(filt(noise(dur), "low", f_hi), "high", f_lo) * shape


def riser(dur):
    x = tt(dur)
    k = x / dur
    sweep = np.sin(2 * np.pi * np.cumsum(180 * 8 ** k) / SR) * k ** 2
    n = filt(noise(dur), "high", 900) * k ** 3
    return sweep * 0.5 + n * 0.8


def heartbeat():
    x = tt(0.6)
    a = np.sin(2 * np.pi * (48 + 30 * np.exp(-x * 20)) * x) * np.exp(-x * 9)
    b = np.zeros_like(x)
    i = int(0.22 * SR)
    b[i:] = a[: len(x) - i] * 0.6
    return a + b


def drone(m, dur):
    s = saw(midi(m), dur, 8, (0, -4, 4)) + 0.6 * np.sin(2 * np.pi * midi(m - 12) * tt(dur))
    return filt(s, "low", 500) * env(dur, 1.5, 99, 1.0, 1.0)


def tick():
    x = tt(0.03)
    return filt(noise(0.03), "band", [2500, 7000]) * np.exp(-x * 300)


# ---------------- 真实音效 ----------------

@lru_cache(maxsize=64)
def _decode(rel):
    p = ASSETS / rel
    if not p.exists():
        return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).astype(np.float64)
    return x / (np.abs(x).max() + 1e-9)


def real(rel, dur=None, fade=0.05, start=0.0):
    x = _decode(rel)
    if x is None:
        return None
    x = x[int(start * SR):].copy()
    if dur is not None:
        n = int(dur * SR)
        x = np.tile(x, n // max(len(x), 1) + 1)[:n] if len(x) < n else x[:n]
    f = min(int(fade * SR), len(x) // 4)
    if f:
        x[:f] *= np.linspace(0, 1, f)
        x[-f:] *= np.linspace(1, 0, f)
    return x


def real_loudest(rel, dur):
    x = _decode(rel)
    if x is None:
        return None
    n = int(dur * SR)
    if len(x) <= n:
        return x.copy()
    e = uniform_filter1d(uniform_filter1d(x ** 2, int(0.5 * SR)), n)
    k = int(np.argmax(e[n // 2: len(x) - n // 2]))
    seg = x[k:k + n].copy()
    f = int(0.05 * SR)
    seg[:f] *= np.linspace(0, 1, f)
    return seg / (np.abs(seg).max() + 1e-9)


# ---------------- 总谱 ----------------

class Score:
    def __init__(self, duration):
        self.duration = duration
        self.N = int((duration + 4) * SR)
        self.bgm = np.zeros((self.N, 2))
        self.sfx = np.zeros((self.N, 2))
        self.dry_bgm = np.zeros((self.N, 2))   # 不进混响的配乐（如合成的低频）
        self.silences = []
        self.gates = []

    def add(self, bus, sig, at, gain=1.0, pan=0.0):
        if sig is None:
            return
        buf = {"bgm": self.bgm, "sfx": self.sfx, "dry": self.dry_bgm}[bus]
        i = int(at * SR)
        if i < 0:
            sig, i = sig[-i:], 0
        sig = sig[: max(0, self.N - i)]
        left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        buf[i:i + len(sig), 0] += sig * gain * left * 1.414
        buf[i:i + len(sig), 1] += sig * gain * right * 1.414

    def silence(self, t0, t1):
        """完全静音（两条总线都清零）。"""
        self.silences.append((t0, t1))

    def gate(self, t0, t1, level=0.08):
        """压低（如"吸气"空拍）。"""
        self.gates.append((t0, t1, level))

    def master(self, path, bgm_target_db=-1.6, sfx_ref_db=-3.0, ceiling_db=-2.0):
        ir_len = int(2.4 * SR)
        xi = np.arange(ir_len) / SR
        r = np.random.default_rng(11)
        ir = r.standard_normal((ir_len, 2)) * np.exp(-xi * 3.0)[:, None]
        ir[:, 0], ir[:, 1] = filt(ir[:, 0], "low", 5000), filt(ir[:, 1], "low", 5000)
        wet = np.stack([fftconvolve(self.bgm[:, c], ir[:, c])[: self.N] for c in range(2)], axis=1)
        wet *= 0.9 * np.abs(self.bgm).max() / (np.abs(wet).max() + 1e-9)
        bgm = self.bgm + wet * 0.35 + self.dry_bgm
        sfx = self.sfx.copy()

        g = np.ones(self.N)
        ramp = int(0.03 * SR)
        for t0, t1, lvl in self.gates:
            i0, i1 = int(t0 * SR), int(t1 * SR)
            g[i0:i1] = np.minimum(g[i0:i1], lvl)
            g[i0:i0 + ramp] = np.minimum(g[i0:i0 + ramp], np.linspace(1, lvl, len(g[i0:i0 + ramp])))
        fade = int(0.012 * SR)
        for t0, t1 in self.silences:
            i0, i1 = int(t0 * SR), int(t1 * SR)
            g[max(i0 - fade, 0):i0] = np.minimum(g[max(i0 - fade, 0):i0], np.linspace(1, 0, len(g[max(i0 - fade, 0):i0])))
            g[i0:i1] = 0
        bgm *= g[:, None]
        sfx *= g[:, None]

        bgm = leveler(bgm, bgm_target_db) * 0.9
        ref = np.percentile(np.abs(sfx[np.abs(sfx).max(axis=1) > 1e-4]).max(axis=1), 99.9) if np.any(sfx) else 1
        sfx *= 10 ** (sfx_ref_db / 20) / (ref + 1e-9)
        mix = limiter(bgm + sfx, ceiling_db)
        end = int(self.duration * SR)
        mix = mix[:end]
        wavfile.write(path, SR, (mix * 32767).astype(np.int16))
        np.save(str(path).replace(".wav", "_bgm.npy"), bgm[:end].astype(np.float32))
        return mix, bgm[:end]


def leveler(x, target_db=-1.6, floor_db=-42.0, max_gain=40.0):
    """BGM 自动电平：按约 0.3 秒窗口的峰值推到目标附近；低于 floor 的地方（静音、抽空）不提升。"""
    mono = np.abs(x).max(axis=1)
    hop = int(0.05 * SR)
    n = len(mono) // hop
    pk = maximum_filter1d(mono[: n * hop].reshape(n, hop).max(axis=1), 6)
    target, floor = 10 ** (target_db / 20), 10 ** (floor_db / 20)
    gain = np.where(pk > floor, target / np.maximum(pk, 1e-9), 1.0)
    gain = uniform_filter1d(np.clip(gain, 0.02, max_gain), 3)
    g = np.interp(np.arange(len(mono)), np.arange(n) * hop + hop / 2, gain)
    return x * g[:, None]


def limiter(x, ceiling_db=-2.0, lookahead_ms=5.0, release_ms=120.0):
    """前瞻峰值限幅：峰值到来前 5 毫秒开始平滑压低增益，释放 120 毫秒，不产生削波失真。
    上限 -2 dBFS：样点间峰值约高 0.5 dB，真峰值留在 -1.5 dBTP 左右，平台转码后不爆音。"""
    ceil = 10 ** (ceiling_db / 20)
    blk = 48
    n = len(x) // blk + 1
    pad = np.zeros((n * blk - len(x), 2))
    xp = np.vstack([x, pad])
    pk = np.abs(xp).max(axis=1).reshape(n, blk).max(axis=1)
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    la = max(1, int(lookahead_ms / 1000 * SR / blk))
    need = minimum_filter1d(need, size=2 * la + 1, origin=-la)
    rel = 1 - np.exp(-blk / (release_ms / 1000 * SR))
    g = np.empty(n)
    cur = 1.0
    for i in range(n):
        v = need[i]
        cur = v if v < cur else cur + (v - cur) * rel
        g[i] = cur
    gs = np.interp(np.arange(n * blk), np.arange(n) * blk + blk / 2, g)[: len(x)]
    return np.clip(x * gs[:, None], -ceil, ceil)
