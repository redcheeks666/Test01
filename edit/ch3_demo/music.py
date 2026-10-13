"""先驱章节奏样片的配乐与音效，全部用代码合成，输出 out/music.wav。

配器：低音弦乐八分音符固定音型 + 铺底和弦 + 定音鼓 + 铜管短音 + 圆号旋律，
音效：盖章、引信、爆炸、火焰、风声、螺旋桨、笔尖、印刷机、火箭嘶鸣、短波、上升音、音爆、斯普特尼克信号。
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, fftconvolve, sosfilt

from timeline import DURATION, SR, t, t_real

OUT = Path(__file__).resolve().parent / "out"
ASSETS = Path(__file__).resolve().parents[2] / "assets"
rng = np.random.default_rng(7)
N = int(DURATION * SR) + SR
music = np.zeros((N, 2))   # 进混响的乐器
dry = np.zeros((N, 2))     # 不进混响的音效


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def filt(x, kind, f):
    sos = butter(4, f, btype=kind, fs=SR, output="sos")
    return sosfilt(sos, x)


def noise(dur):
    return rng.standard_normal(int(dur * SR))


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def add(buf, sig, at_sec, gain=1.0, pan=0.0):
    i = int(at_sec * SR)
    sig = sig[: max(0, N - i)]
    left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(sig), 0] += sig * gain * left * 1.414
    buf[i:i + len(sig), 1] += sig * gain * right * 1.414


def env(dur, a, d, s_level=0.0, r=0.05):
    """起音 a 秒，衰减 d 秒到持续电平，最后 r 秒释放。"""
    n = int(dur * SR)
    x = tt(dur)
    e = np.where(x < a, x / max(a, 1e-4), s_level + (1 - s_level) * np.exp(-(x - a) / max(d, 1e-4)))
    rn = int(r * SR)
    if rn and n > rn:
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


# ---------- 乐器 ----------

def strings_short(m, dur):
    s = saw(midi(m), dur + 0.15)
    s = filt(s, "low", 1800)
    return s * env(dur + 0.15, 0.012, 0.16, 0.0, 0.08)


def pad(ms, dur):
    s = sum(saw(midi(m), dur, 10, (0, -9, 9)) for m in ms)
    s = filt(s, "low", 1400)
    return s * env(dur, 0.5, 9.0, 1.0, 0.6) / len(ms)


def horn(m, dur):
    x = tt(dur + 0.3)
    f = midi(m) * (1 + 0.004 * np.sin(2 * np.pi * 5 * x) * np.clip(x / 0.4, 0, 1))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = sum(np.sin(k * ph) * a for k, a in [(1, 1), (2, .5), (3, .32), (4, .18), (5, .1), (6, .05)])
    s = filt(s, "low", 2200)
    return s * env(dur + 0.3, 0.08, 3.0, 0.85, 0.3)


def brass_stab(ms, dur=0.5):
    s = sum(saw(midi(m), dur, 18, (0, -5, 5)) for m in ms)
    s = filt(s, "low", 3200)
    return s * env(dur, 0.006, 0.18, 0.0, 0.1) / len(ms)


def timpani(m, dur=1.6, vel=1.0):
    x = tt(dur)
    f0 = midi(m)
    f = f0 * (1 + 0.06 * np.exp(-x * 25))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-x * 2.2) + 0.5 * np.sin(1.5 * ph) * np.exp(-x * 5) + 0.3 * np.sin(1.99 * ph) * np.exp(-x * 7)
    hit = filt(noise(dur), "low", 900) * np.exp(-x * 60) * 0.6
    return (s + hit) * vel


def snare(vel=1.0):
    d = 0.18
    x = tt(d)
    n = filt(noise(d), "band", [1200, 6000]) * np.exp(-x * 28)
    body = np.sin(2 * np.pi * 190 * x) * np.exp(-x * 40) * 0.5
    return (n + body) * vel


def cymbal(dur=2.5):
    x = tt(dur)
    return filt(noise(dur), "high", 5000) * np.exp(-x * 1.8)


# ---------- 音效 ----------

def stamp():
    x = tt(0.35)
    return np.sin(2 * np.pi * 110 * x) * np.exp(-x * 22) + filt(noise(0.35), "low", 1500) * np.exp(-x * 40) * 0.6


def fuse(dur):
    x = tt(dur)
    hiss = filt(noise(dur), "high", 4500) * (0.6 + 0.4 * np.sin(2 * np.pi * 9 * x))
    crackle = (rng.random(len(x)) > 0.9985) * rng.uniform(0.5, 1.5, len(x))
    crackle = filt(crackle, "high", 2000) * 3
    return (hiss * 0.35 + crackle) * np.clip(x / 0.15, 0, 1)


def boom(dur=4.0, f0=60, f1=26, crack=1.0):
    x = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-x * 2.5)
    low = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 1.1)
    rumble = filt(noise(dur), "low", 220) * np.exp(-x * 1.4) * 1.6
    cr = filt(noise(dur), "high", 1500) * np.exp(-x * 30) * crack
    return np.tanh((low * 1.2 + rumble + cr) * 1.4)


def whoosh(dur=0.7, f_lo=300, f_hi=3000):
    x = tt(dur)
    shape = np.sin(np.pi * np.clip(x / dur, 0, 1)) ** 2
    return filt(filt(noise(dur), "low", f_hi), "high", f_lo) * shape


def wind(dur):
    x = tt(dur)
    s = filt(noise(dur), "band", [200, 1400]) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * x + 1))
    return s * env(dur, 0.3, 99, 1.0, 0.25)


def fire(dur):
    x = tt(dur)
    return filt(noise(dur), "low", 700) * (0.7 + 0.3 * np.sin(2 * np.pi * 3.1 * x)) * env(dur, 0.1, 99, 1.0, 0.4)


def propeller(dur):
    x = tt(dur)
    s = saw(42, dur, 25, (0,)) * (0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 17 * x)))
    return filt(s, "low", 900) * env(dur, 0.25, 99, 1.0, 0.3)


def pen(dur):
    x = tt(dur)
    am = np.clip(np.sin(2 * np.pi * 6.3 * x) + 0.3 * np.sin(2 * np.pi * 11 * x), 0, None)
    return filt(noise(dur), "band", [2500, 7000]) * am


def press(dur, beat_len):
    """印刷机：十六分音符的金属咔嗒 + 每拍一次低沉的滚筒声。"""
    out = np.zeros(int(dur * SR))
    step = beat_len / 4
    k = 0
    while k * step < dur - 0.05:
        i = int(k * step * SR)
        d = 0.06
        x = tt(d)
        click = filt(noise(d), "band", [1800, 6500]) * np.exp(-x * 90) * (1.0 if k % 2 == 0 else 0.6)
        out[i:i + len(click)] += click[: len(out) - i]
        if k % 4 == 0:
            d2 = 0.2
            x2 = tt(d2)
            th = np.sin(2 * np.pi * 70 * x2) * np.exp(-x2 * 25)
            out[i:i + len(th)] += th[: len(out) - i]
        k += 1
    return out


def rocket_hiss(dur):
    x = tt(dur)
    return filt(noise(dur), "band", [400, 5000]) * np.clip(x / 0.3, 0, 1) * np.exp(-np.clip(x - dur + 0.4, 0, None) * 8)


def shortwave(dur):
    x = tt(dur)
    s = filt(noise(dur), "band", [500, 3200]) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * x) ** 2)
    whistle = np.sin(2 * np.pi * (900 + 300 * np.sin(2 * np.pi * 0.25 * x)) * x) * 0.15
    return (s * 0.5 + whistle) * env(dur, 0.4, 99, 1.0, 0.3)


def riser(dur):
    x = tt(dur)
    k = x / dur
    sweep = np.sin(2 * np.pi * np.cumsum(180 * 8 ** k) / SR) * k ** 2
    n = filt(noise(dur), "high", 900) * k ** 3
    return sweep * 0.5 + n * 0.8


def sputnik_beep():
    d = 0.3
    x = tt(d)
    s = np.sign(np.sin(2 * np.pi * 1180 * x)) * 0.5 + np.sin(2 * np.pi * 1180 * x)
    s = filt(s, "low", 3500)
    e = np.clip(x / 0.01, 0, 1) * np.clip((d - x) / 0.02, 0, 1)
    return s * e


# ---------- 真实音效（assets/ 原件，缺失时退回合成） ----------

def real(rel, dur=None, fade=0.05):
    """读取 assets/ 下的音频原件，解码为 48k 单声道，峰值归一化；dur 给定时截断或循环到该长度。"""
    import subprocess
    p = ASSETS / rel
    if not p.exists():
        return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).astype(np.float64)
    x /= np.abs(x).max() + 1e-9
    if dur is not None:
        n = int(dur * SR)
        x = np.tile(x, n // len(x) + 1)[:n] if len(x) < n else x[:n]
    f = min(int(fade * SR), len(x) // 4)
    if f:
        x[:f] *= np.linspace(0, 1, f)
        x[-f:] *= np.linspace(1, 0, f)
    return x


def pick(real_sig, synth_sig):
    return real_sig if real_sig is not None else synth_sig


def keys(text_len, at_sec, gain=0.12):
    """日期戳逐字打出时的键盘声：每两个字符一下。"""
    k = real("FX/FX-01/FX-01_01_mech_keyboard_click_pixabay.mp3", fade=0.005)
    if k is None:
        return
    for i in range(0, text_len, 2):
        add(dry, k * rng.uniform(0.7, 1.0), at_sec + i / 30, gain, pan=rng.uniform(-0.3, 0.3))


# ---------- 编曲 ----------

def bl(b0, b1):
    return t(b1) - t(b0)


ROOTS = {"Dm": 38, "Bb": 34, "F": 41, "C": 36, "Gm": 43, "A": 33}   # 低音区根音（MIDI）
TRIADS = {"Dm": [50, 53, 57], "Bb": [46, 50, 53], "F": [45, 48, 53], "C": [48, 52, 55],
          "Gm": [43, 50, 55], "A": [45, 49, 52]}
QUALITY = {"Dm": 3, "Bb": 4, "F": 4, "C": 4, "Gm": 3, "A": 4}
# 每小节的和弦（每小节 4 拍，从第 0 拍开始）
BARS = ["Dm", "Dm", "Dm", "Dm", "Bb", "F", "C", "A", "Dm", "Bb", "Gm", "A", "Dm", "Dm", "Dm"]


def ostinato(b_from, b_to, gain=0.35):
    """八分音符低音弦乐固定音型：根、根、五、根、八度、根、五、小七/大六。"""
    b = b_from
    while b < b_to - 1e-6:
        ch = BARS[int(b // 4)]
        r = ROOTS[ch]
        pattern = [0, 0, 7, 0, 12, 0, 7, 10 if QUALITY[ch] == 3 else 9]
        m = r + pattern[int(round((b % 4) * 2)) % 8]
        dur = bl(b, b + 0.5) * 0.8
        add(music, strings_short(m, dur), t(b), gain * (1.15 if b % 1 == 0 else 0.85), pan=-0.25)
        add(music, strings_short(m + 12, dur), t(b), gain * 0.35, pan=0.3)
        b += 0.5


def pads(b_from, b_to, gain=0.22):
    b = b_from
    while b < b_to - 1e-6:
        nxt = min((int(b // 4) + 1) * 4, b_to)
        add(music, pad(TRIADS[BARS[int(b // 4)]], bl(b, nxt) + 0.4), t(b), gain)
        b = nxt


def timp_pattern(b_from, b_to, every=2.0, gain=0.7):
    b = b_from
    while b < b_to - 1e-6:
        ch = BARS[int(b // 4)]
        m = ROOTS[ch] if b % 4 == 0 else ROOTS[ch] + 7 - 12
        add(music, timpani(m, vel=1.0 if b % 4 == 0 else 0.7), t(b), gain)
        b += every


def stab(b, gain=0.5):
    ch = BARS[int(b // 4)]
    add(music, brass_stab([m + 12 for m in TRIADS[ch]] + [ROOTS[ch] + 12]), t(b), gain)


def real_loudest(rel, dur):
    """取音频原件里最响的一段（按 0.5 秒窗口的 RMS）。"""
    x = real(rel, fade=0.0)
    if x is None:
        return None
    win = int(0.5 * SR)
    e = uniform_filter1d(x ** 2, win)
    n = int(dur * SR)
    if len(x) <= n:
        return x
    k = int(np.argmax(uniform_filter1d(e, n)[n // 2: len(x) - n // 2])) if len(x) > n else 0
    seg = x[k:k + n].copy()
    f = int(0.05 * SR)
    seg[:f] *= np.linspace(0, 1, f)
    return seg / (np.abs(seg).max() + 1e-9)


def ostinato_real(r0, r1, ch, gain=0.35):
    """按实际拍号写的固定音型（万户段第 16—20 拍用）。"""
    r = r0
    root = ROOTS[ch]
    pattern = [0, 0, 7, 0, 12, 0, 7, 10 if QUALITY[ch] == 3 else 9]
    while r < r1 - 1e-6:
        m = root + pattern[int(round((r % 4) * 2)) % 8]
        dur = (t_real(r + 0.5) - t_real(r)) * 0.8
        add(music, strings_short(m, dur), t_real(r), gain * (1.2 if r % 1 == 0 else 0.9), pan=-0.25)
        add(music, strings_short(m + 12, dur), t_real(r), gain * 0.4, pan=0.3)
        r += 0.5


def ignition(r_fire, r_end):
    """万户点火起飞：真实发射轰鸣 + 定音鼓与铜管重音 + 属和弦固定音型 + 小军鼓滚奏推到下一小节。"""
    t0, t1 = t_real(r_fire), t_real(r_end)
    roar = real_loudest("FX/FX-02/FX-02_01_falcon9_pad_mics_nasa.wav", t1 - t0 + 1.0)
    if roar is not None:
        x = np.arange(len(roar)) / SR
        shape = np.clip(x / 0.08, 0, 1) * (0.75 + 0.25 * np.clip(x / 1.5, 0, 1))
        shape *= np.clip((t1 - t0 + 0.6 - x) / 0.8, 0, 1)                    # 飞出画面后渐远
        add(dry, filt(roar, "low", 6000) * shape, t0, 0.8)
    add(dry, boom(1.6, 48, 30, 0.2), t0, 0.35)                                # 点火瞬间的低频
    lh = real("FX/FX-11/FX-11_02_cinematic_low_hit_pixabay.mp3")
    if lh is not None:
        add(dry, lh, t0, 0.5)
    add(music, timpani(33, vel=1.3), t0, 0.9)
    add(music, brass_stab([69, 73, 76, 81, 57], 0.9), t0, 0.6)
    add(music, cymbal(2.0), t0, 0.22)
    ostinato_real(r_fire, r_end, "A", 0.38)
    add(music, pad([45, 52, 57, 61, 64], t1 - t0 + 0.3), t0, 0.45)
    for k, m in enumerate([69, 71, 73, 76]):          # 圆号逐拍上行，把起飞推上去
        rr = r_fire + 1 + k * 0.75
        add(music, horn(m, t_real(rr + 0.75) - t_real(rr)), t_real(rr), 0.35, pan=0.15)
    r = r_fire + 2.5
    while r < r_end - 0.1:
        k = (r - r_fire - 2.5) / (r_end - r_fire - 2.5)
        add(music, snare(0.3 + 0.7 * k), t_real(r), 0.32)
        r += 0.25 if r < r_end - 0.75 else 0.125
    up = real("FX/FX-11/FX-11_01_short_whoosh_pixabay.mp3")
    add(dry, pick(up, whoosh(0.6, 400, 5000)), t_real(19.2), 0.5)


def bgm_leveler(x, target_db=-1.6, floor_db=-40.0):
    """BGM 自动电平：按 0.4 秒窗口的峰值把配乐推到目标电平附近（保持在 0 到 −6 dBFS 之间），
    低于 floor_db 的地方（刻意的静音、抽空）不提升。"""
    mono = np.abs(x).max(axis=1)
    hop = int(0.05 * SR)
    n = len(mono) // hop
    pk = mono[: n * hop].reshape(n, hop).max(axis=1)
    pk = maximum_filter1d(pk, 6)                       # 0.3 秒
    target = 10 ** (target_db / 20)
    floor = 10 ** (floor_db / 20)
    gain = np.where(pk > floor, target / np.maximum(pk, 1e-9), 1.0)
    gain = np.clip(gain, 0.02, 40.0)
    gain = uniform_filter1d(gain, 3)                   # 0.15 秒平滑
    g = np.interp(np.arange(len(mono)), np.arange(n) * hop + hop / 2, gain)
    return x * g[:, None]


def main():
    # 3-01 章节卡（0—8）：低音铺底 + 定音鼓滚奏渐强 + 盖章
    add(music, pad([38, 45, 50], bl(0, 8) + 0.2), t(0), 0.18)
    b = 4.0
    while b < 7.75:
        k = (b - 4) / 4
        add(music, timpani(38, 0.6, 0.25 + 0.75 * k), t(b), 0.6)
        b += 0.25
    add(dry, stamp(), t(6), 0.8)
    add(dry, pick(real("FX/FX-11/FX-11_01_short_whoosh_pixabay.mp3"), whoosh(bl(7.5, 8) + 0.05, 500, 5000)), t(7.5), 0.45)

    # 3-02 万户（实际 8—20 拍）：固定音型进入，引信嘶嘶，第 15.5 拍抽空；
    # 第 16 拍点火——真实火箭发射轰鸣 + 定音鼓 + 铜管，属和弦持续推高；第 17 拍离地，第 19.4 拍飞出画面
    ostinato(8, 15.5)
    pads(8, 15.5, 0.16)
    timp_pattern(8, 15.5)
    add(dry, pick(real("FX/FX-08/FX-08_01_fuse_burning_pixabay.mp3", t_real(13) - t_real(8)), fuse(t_real(13) - t_real(8))),
        t_real(8), 0.3, pan=0.2)
    add(dry, fuse(t_real(15.5) - t_real(13)) * 0.6, t_real(13), 0.25, pan=0.3)   # 喷口零星的火星
    stab(8, 0.45)
    ignition(16, 20)

    # 3-03 热气球（逻辑 16—24）：全编制 + 旋律
    stab(16, 0.6)
    add(music, timpani(38, vel=1.2), t(16), 0.9)
    ostinato(16, 44)
    pads(16, 44)
    timp_pattern(18, 44)
    add(dry, pick(real("FX/FX-07/FX-07_03_fire_sounds_pixabay.mp3", bl(16.5, 22), 0.4), fire(bl(16.5, 22))), t(16.5), 0.22, pan=-0.3)
    melody = [(16, 62, 2), (18, 65, 1), (19, 69, 1), (20, 67, 2), (22, 65, 1), (23, 64, 1),
              (24, 69, 2), (26, 72, 1), (27, 74, 1), (28, 72, 2), (30, 69, 2),
              (32, 67, 2), (34, 70, 1), (35, 69, 1), (36, 69, 4),
              (40, 74, 2), (42, 72, 1), (43, 73, 1)]
    for b0, m, d in melody:
        add(music, horn(m, bl(b0, b0 + d) * 0.95), t(b0), 0.2, pan=0.15)

    # 3-06 李林塔尔（24—28）：甩入 + 风声，最后半拍风声骤停
    add(dry, pick(real("FX/FX-11/FX-11_01_short_whoosh_pixabay.mp3"), whoosh(0.5, 300, 4000)), t(24) - 0.25, 0.5)
    add(dry, pick(real("FX/FX-07/FX-07_02_harsh_wind_pixabay.mp3", bl(24, 27.5), 0.2), wind(bl(24, 27.5))), t(24), 0.35)

    # 3-07 1903（28—36）：铜管重音 + 螺旋桨 + 笔尖
    stab(28, 0.6)
    add(music, cymbal(1.5), t(28), 0.15)
    add(dry, propeller(bl(28, 34)), t(28), 0.22, pan=-0.6)
    add(dry, pen(bl(28.5, 33.5)), t(28.5), 0.12, pan=0.6)
    for b0 in (30, 32):
        add(dry, whoosh(0.18, 800, 6000), t(b0) - 0.06, 0.2)

    # 3-08 纽约时报（36—40）：印刷机 + 两次重音
    add(dry, press(bl(36, 40), bl(36, 37)), t(36), 0.3)
    stab(36, 0.6)
    stab(38, 0.5)
    add(dry, pick(real("FX/FX-11/FX-11_01_short_whoosh_pixabay.mp3"), whoosh(0.3, 400, 5000)), t(38) - 0.15, 0.4)

    # 3-09 戈达德（40—44）：定音鼓八分音符 + 2.5 秒火箭嘶鸣
    timp_pattern(40, 44, every=0.5, gain=0.45)
    add(dry, rocket_hiss(2.5), t(40.5), 0.35)
    add(music, cymbal(1.2), t(40), 0.2)

    # 3-11 V-2（44—48）：固定音型和鼓撤掉（44 之后不再写入），只留铺底 + 短波
    add(dry, pick(real("FX/FX-09/FX-09_02_shortwave_tuning_noise_pixabay.mp3", bl(44, 49), 0.4), shortwave(bl(44, 49))), t(44), 0.22)
    add(music, pad([50, 57, 62], bl(44, 50)), t(44), 0.12)

    # 3-12 X-1（48—52）：上升音 + 小军鼓加速滚奏 → 第 50 拍音爆
    add(dry, riser(bl(46, 49.85)), t(46), 0.45)
    b = 48.0
    while b < 49.85:
        k = (b - 48) / 2
        add(music, snare(0.3 + 0.7 * k), t(b), 0.35)
        b += 0.25 if b < 49 else 0.125
    add(dry, boom(3.0, 55, 24, 2.0), t(50), 0.9)
    lh = real("FX/FX-11/FX-11_02_cinematic_low_hit_pixabay.mp3")
    if lh is not None:
        add(dry, lh, t(50), 0.7)
    add(music, timpani(38, vel=1.3), t(50), 0.9)
    add(music, cymbal(2.0), t(50), 0.3)
    stab(50, 0.7)

    # 日期戳的打字声
    for b0, n in [(17, 18), (24.5, 9), (36.3, 10), (40.3, 10), (44.8, 10)]:
        keys(n, t(b0))

    # 混响
    ir_len = int(2.2 * SR)
    xi = np.arange(ir_len) / SR
    ir = rng.standard_normal((ir_len, 2)) * np.exp(-xi * 3.2)[:, None]
    ir[:, 0], ir[:, 1] = filt(ir[:, 0], "low", 5000), filt(ir[:, 1], "low", 5000)
    wet = np.stack([fftconvolve(music[:, c], ir[:, c])[:N] for c in range(2)], axis=1)
    wet *= 0.9 / (np.abs(wet).max() + 1e-9) * np.abs(music).max()
    bgm = music + wet * 0.35
    sfx = dry.copy()

    # 第 15.5 拍"吸气"：把混响尾音压下去，让爆炸前真正空一下
    g0, g1 = int(t(15.5) * SR), int(t(16) * SR) - int(0.004 * SR)
    ramp = int(0.03 * SR)
    gate = np.ones(N)
    gate[g0:g1] = 0.08
    gate[g0:g0 + ramp] = np.linspace(1, 0.08, ramp)
    bgm *= gate[:, None]
    sfx *= gate[:, None]

    # 第 52 拍起一拍完全静音，之后只有斯普特尼克信号和极弱的短波底噪
    i52, i53 = int(t(52) * SR), int(t(53) * SR)
    fade = int(0.012 * SR)
    for bus in (bgm, sfx):
        bus[i52 - fade:i52] *= np.linspace(1, 0, fade)[:, None]
        bus[i52:] = 0
    # 第 53 拍：下一章章节卡随重音落下——定音鼓 + 铜管大三和弦 + 低频冲击，弦乐长音收尾；第 55.2 拍盖印
    tail = np.zeros((N, 2))
    tail_music = np.zeros((N, 2))
    b53 = t(53)
    add(tail_music, timpani(38, 2.4, 1.3), b53, 0.9)
    add(tail_music, brass_stab([62, 66, 69, 74, 50], 1.2), b53, 0.65)
    add(tail_music, pad([50, 57, 62, 66, 69], DURATION - b53), b53, 0.3)
    add(tail_music, cymbal(2.5), b53, 0.2)
    lh = real("FX/FX-11/FX-11_02_cinematic_low_hit_pixabay.mp3")
    if lh is not None:
        add(tail, lh, b53, 0.6)
    add(tail, stamp(), t(55.2), 0.7)
    wet2 = np.stack([fftconvolve(tail_music[:, c], ir[:, c])[:N] for c in range(2)], axis=1)
    wet2 *= 0.9 / (np.abs(wet2).max() + 1e-9) * np.abs(tail_music).max()
    n_end = int(DURATION * SR)
    fade_len = int(1.2 * SR)
    end_fade = np.ones(N)
    end_fade[n_end - fade_len:n_end] = np.linspace(1, 0, fade_len)
    bgm += (tail_music + wet2 * 0.25) * end_fade[:, None]
    sfx += tail * end_fade[:, None]

    # BGM 全程拉到 0 到 −6 dBFS；音效整体按 BGM 配比：最响的音效峰值约 −2 dBFS
    bgm = bgm_leveler(bgm)
    sfx *= 10 ** (-2 / 20) / (np.abs(sfx).max() + 1e-9)
    mix = bgm * 0.9 + sfx
    # 软限幅，峰值不超过 −0.5 dBFS
    ceil = 10 ** (-0.5 / 20)
    mix = np.tanh(mix / ceil) * ceil
    mix = mix[: int(DURATION * SR)]
    np.save(OUT / "bgm_bus.npy", (bgm * 0.9)[: int(DURATION * SR)].astype(np.float32))
    OUT.mkdir(exist_ok=True)
    wavfile.write(OUT / "music.wav", SR, (mix * 32767).astype(np.int16))
    print("music.wav", round(DURATION, 2), "s, peak", round(float(np.abs(mix).max()), 3))


if __name__ == "__main__":
    main()
