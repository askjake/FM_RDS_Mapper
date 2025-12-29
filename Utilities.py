# -*- coding: utf-8 -*-
"""
Utilities.py (v31)

Small helper utilities (events/logging, DSP primitives, gain quantization, tool resolution).
Deliberately avoids heavy dependencies (numpy only).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from Defaults import (
    HACKRF_LNA_STEPS_DB,
    HACKRF_VGA_MIN_DB,
    HACKRF_VGA_MAX_DB,
    HACKRF_VGA_STEP_DB,
    PILOT_HZ,
    RDS_SUBCARRIER_HZ,
)

# ==========================
# Time / events
# ==========================
def utc_iso(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")

def emit_event(q, event: Dict[str, Any]) -> None:
    """Best-effort queue emit (never throws)."""
    if q is None:
        return
    try:
        if "ts" not in event:
            event["ts"] = utc_iso()
        q.put_nowait(event)
    except Exception:
        pass

def tail_lines(text: str, n: int = 25) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()
    return lines[-int(n):]

# ==========================
# Tool resolution
# ==========================
def which_or_none(path_or_name: str) -> Optional[str]:
    if not path_or_name:
        return None
    # absolute / relative file path
    if os.path.exists(path_or_name):
        return os.path.abspath(path_or_name)
    # PATH lookup
    hit = shutil.which(path_or_name)
    return hit

def resolve_tool(path_or_name: str, fallbacks: Sequence[str] = ()) -> Optional[str]:
    hit = which_or_none(path_or_name)
    if hit:
        return hit
    for fb in fallbacks:
        hit = which_or_none(fb)
        if hit:
            return hit
    return None

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

# ==========================
# Gain quantization
# ==========================
def quantize_hackrf_gain(lna_db: int, vga_db: int) -> Tuple[int, int, bool]:
    """
    Quantize gain to HackRF-supported steps.
    Returns (lna_applied, vga_applied, changed)
    """
    lna_db = int(lna_db)
    vga_db = int(vga_db)

    lna_applied = min(HACKRF_LNA_STEPS_DB, key=lambda s: abs(int(s) - lna_db))
    v = int(round(vga_db / HACKRF_VGA_STEP_DB) * HACKRF_VGA_STEP_DB)
    vga_applied = max(HACKRF_VGA_MIN_DB, min(HACKRF_VGA_MAX_DB, v))

    changed = (lna_applied != lna_db) or (vga_applied != vga_db)
    return int(lna_applied), int(vga_applied), bool(changed)

# ==========================
# DSP helpers
# ==========================
def fir_lowpass(numtaps: int, cutoff_hz: float, fs: float) -> np.ndarray:
    """
    Windowed-sinc lowpass FIR (Hann window).
    cutoff_hz must be < fs/2.
    """
    numtaps = int(numtaps)
    if numtaps < 7:
        numtaps = 7
    if numtaps % 2 == 0:
        numtaps += 1

    cutoff_hz = float(cutoff_hz)
    fs = float(fs)
    cutoff_hz = min(cutoff_hz, 0.499 * fs)

    fc = cutoff_hz / fs  # cycles per sample (0..0.5)
    n = np.arange(numtaps, dtype=np.float32) - (numtaps - 1) / 2.0
    # numpy.sinc(x) = sin(pi*x)/(pi*x)
    h = 2.0 * fc * np.sinc(2.0 * fc * n)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(numtaps, dtype=np.float32) / (numtaps - 1)))
    h = h * w
    h = h / (np.sum(h) + 1e-12)
    return h.astype(np.float32)

def fir_bandpass(numtaps: int, f1_hz: float, f2_hz: float, fs: float) -> np.ndarray:
    f1_hz = float(f1_hz)
    f2_hz = float(f2_hz)
    if f2_hz <= f1_hz:
        raise ValueError("bandpass f2 must be > f1")
    lp2 = fir_lowpass(numtaps, f2_hz, fs)
    lp1 = fir_lowpass(numtaps, f1_hz, fs)
    bp = lp2 - lp1
    bp = bp / (np.sum(np.abs(bp)) + 1e-12)
    return bp.astype(np.float32)

@dataclass
class FIRDecimator:
    """
    Streaming FIR + decimation for complex IQ.
    Keeps filter state and a global decimation phase.

    Implemented using np.convolve on each chunk (simple + robust).
    """
    taps: np.ndarray
    decim: int

    def __post_init__(self):
        self.taps = np.asarray(self.taps, dtype=np.float32)
        self.decim = int(self.decim)
        self._zi = np.zeros(len(self.taps) - 1, dtype=np.complex64)
        self._total_in = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        if x is None or x.size == 0:
            return np.empty((0,), dtype=np.complex64)
        x = x.astype(np.complex64, copy=False)
        x_ext = np.concatenate([self._zi, x])
        y = np.convolve(x_ext, self.taps, mode="valid").astype(np.complex64)
        start = (-self._total_in) % self.decim
        y_dec = y[start::self.decim]
        self._total_in += int(x.size)
        self._zi = x_ext[-(len(self.taps) - 1):].astype(np.complex64, copy=True)
        return y_dec

@dataclass
class MovingRMS:
    """Exponentially-smoothed RMS tracker for float signals."""
    alpha: float = 0.02
    value: float = 0.0

    def update(self, x: np.ndarray) -> float:
        if x is None or x.size == 0:
            return float(self.value)
        rms = float(np.sqrt(np.mean(np.square(x.astype(np.float32)))))
        self.value = (1.0 - float(self.alpha)) * float(self.value) + float(self.alpha) * rms
        return float(self.value)

def fm_quadrature_demod(x: np.ndarray, prev: Optional[np.complex64] = None) -> Tuple[np.ndarray, Optional[np.complex64]]:
    """
    Quadrature FM demod for complex baseband.
    Returns (y, last_sample) so callers can stitch chunk boundaries.

    y is in radians/sample; absolute scaling isn't critical for redsea.
    """
    if x is None or x.size == 0:
        return np.empty((0,), dtype=np.float32), prev
    x = x.astype(np.complex64, copy=False)

    if prev is not None:
        x2 = np.concatenate([np.asarray([prev], dtype=np.complex64), x])
    else:
        x2 = x

    if x2.size < 2:
        return np.empty((0,), dtype=np.float32), x2[-1] if x2.size else prev

    y = np.angle(x2[1:] * np.conj(x2[:-1])).astype(np.float32)
    return y, x2[-1]

def float_to_int16_pcm(x: np.ndarray, percentile: float = 99.9) -> Tuple[np.ndarray, float, float]:
    """
    Scale float signal to int16 using a robust peak estimate.
    Returns (pcm_int16, scale, clip_pct)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros((0,), dtype=np.int16), 1.0, 0.0
    peak = float(np.percentile(np.abs(x), float(percentile))) + 1e-9
    scale = (0.95 * 32767.0) / peak
    y = x * scale
    clip = float(np.mean(np.abs(y) >= 32767.0) * 100.0)
    y = np.clip(y, -32767.0, 32767.0)
    return y.astype(np.int16), float(scale), float(clip)

def estimate_pilot_and_rds_snr_db(mpx: np.ndarray, fs: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Rough SNR estimates using FFT bins:
    - pilot @ 19 kHz
    - rds energy around 57 kHz +/- 2.4 kHz

    Returns (pilot_snr_db, rds_snr_db). None if insufficient data.
    """
    mpx = np.asarray(mpx, dtype=np.float32)
    fs = float(fs)
    if mpx.size < int(fs * 0.5):
        return None, None

    n = int(fs)  # 1 second
    seg = mpx[-n:] if mpx.size >= n else mpx
    seg = seg - float(np.mean(seg))
    win = np.hanning(seg.size).astype(np.float32)
    X = np.fft.rfft(seg * win)
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / fs)

    def band_mean(f_lo: float, f_hi: float) -> float:
        idx = np.where((freqs >= float(f_lo)) & (freqs <= float(f_hi)))[0]
        if idx.size == 0:
            return 0.0
        return float(np.mean(P[idx]))

    pilot_p = band_mean(PILOT_HZ - 200.0, PILOT_HZ + 200.0)
    pilot_n = band_mean(PILOT_HZ + 600.0, PILOT_HZ + 2400.0) + band_mean(PILOT_HZ - 2400.0, PILOT_HZ - 600.0)
    pilot_snr = 10.0 * math.log10((pilot_p + 1e-12) / (pilot_n + 1e-12))

    bw = 2400.0
    rds_p = band_mean(RDS_SUBCARRIER_HZ - bw, RDS_SUBCARRIER_HZ + bw)
    rds_n = band_mean(RDS_SUBCARRIER_HZ + 5000.0, RDS_SUBCARRIER_HZ + 12000.0) + band_mean(RDS_SUBCARRIER_HZ - 12000.0, RDS_SUBCARRIER_HZ - 5000.0)
    rds_snr = 10.0 * math.log10((rds_p + 1e-12) / (rds_n + 1e-12))

    pilot_snr = float(max(-120.0, min(60.0, pilot_snr)))
    rds_snr = float(max(-120.0, min(60.0, rds_snr)))
    return pilot_snr, rds_snr

def parse_json_lines(text: str) -> Iterable[Dict[str, Any]]:
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj
