"""先驱章节奏样片的拍子网格：96 BPM 在前 44 拍渐快到 112 BPM，之后保持。

万户段（实际第 8—20 拍）比其他镜头多一小节，用来做点火、离地、飞出画面。
其后各镜头仍按原来的"逻辑拍号"编写（热气球从逻辑第 16 拍开始），
t(逻辑拍) 会自动顺延 4 拍换算成实际时间；万户段内部用 t_real(实际拍)。
"""
import numpy as np

FPS = 30
SR = 48000
WANHU_END = 20          # 万户段结束的实际拍号
SHIFT = 4               # 逻辑第 16 拍及以后顺延的拍数
TOTAL_BEATS = 64        # 实际拍数
RAMP_BEATS = 44
BPM0, BPM1 = 96.0, 112.0


def bpm(beat):
    return BPM0 + (BPM1 - BPM0) * min(max(beat, 0) / RAMP_BEATS, 1.0)


# 以 1/64 拍为步长积分，得到拍号 → 时间的对照表
_STEP = 1 / 64
_grid = np.arange(0, TOTAL_BEATS + 4, _STEP)
_times = np.concatenate([[0], np.cumsum(60.0 / np.array([bpm(b) for b in _grid[:-1]]) * _STEP)])


def t_real(beat):
    """实际拍号 → 秒。"""
    return float(np.interp(beat, _grid, _times))


def to_real(logical):
    return logical if logical < 16 else logical + SHIFT


def t(logical):
    """逻辑拍号 → 秒（第 16 拍及以后顺延 SHIFT 拍）。"""
    return t_real(to_real(logical))


def beat_at(sec):
    """秒 → 实际拍号。"""
    return float(np.interp(sec, _times, _grid))


def frame(logical):
    return int(round(t(logical) * FPS))


def frame_real(beat):
    return int(round(t_real(beat) * FPS))


DURATION = t_real(TOTAL_BEATS) + 0.6
