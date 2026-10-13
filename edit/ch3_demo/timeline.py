"""先驱章节奏样片的拍子网格：96 BPM 在前 40 拍渐快到 112 BPM，之后保持。"""
import numpy as np

FPS = 30
SR = 48000
TOTAL_BEATS = 60
RAMP_BEATS = 40
BPM0, BPM1 = 96.0, 112.0


def bpm(beat):
    return BPM0 + (BPM1 - BPM0) * min(max(beat, 0) / RAMP_BEATS, 1.0)


# 以 1/64 拍为步长积分，得到拍号 → 时间的对照表
_STEP = 1 / 64
_grid = np.arange(0, TOTAL_BEATS + 4, _STEP)
_times = np.concatenate([[0], np.cumsum(60.0 / np.array([bpm(b) for b in _grid[:-1]]) * _STEP)])


def t(beat):
    """拍号（可为小数）→ 秒。"""
    return float(np.interp(beat, _grid, _times))


def beat_at(sec):
    """秒 → 拍号。"""
    return float(np.interp(sec, _times, _grid))


def frame(beat):
    return int(round(t(beat) * FPS))


DURATION = t(TOTAL_BEATS) + 0.6
