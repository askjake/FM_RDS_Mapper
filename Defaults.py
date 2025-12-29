# -*- coding: utf-8 -*-
"""
Defaults.py (v31)

Central place for default parameters and small dataclasses shared across the app.
Keep this file dependency-light.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# ==========================
# Band / tuning defaults
# ==========================
FM_MIN_MHZ_DEFAULT = 88.0
FM_MAX_MHZ_DEFAULT = 108.0
FM_STEP_KHZ_DEFAULT = 200

# ==========================
# HackRF constraints / steps
# ==========================
# HackRF: lowest practical sample rate for FM is usually 2 Msps+
HACKRF_MIN_SAMPLE_RATE_HZ = 2_000_000
HACKRF_MAX_SAMPLE_RATE_HZ = 20_000_000

# hackrf_transfer gain steps
HACKRF_LNA_STEPS_DB = (0, 8, 16, 24, 32, 40)  # max 40 dB
HACKRF_VGA_MIN_DB = 0
HACKRF_VGA_MAX_DB = 62
HACKRF_VGA_STEP_DB = 2

# FM/RDS basics
PILOT_HZ = 19_000
RDS_SUBCARRIER_HZ = 57_000

# ==========================
# Default tool paths
# ==========================
@dataclass
class ToolPaths:
    """
    Tool paths can be absolute (recommended on Windows) or just names in PATH.
    The app will try to resolve them.
    """
    hackrf_transfer: str = "hackrf_transfer"
    hackrf_sweep: str = "hackrf_sweep"
    redsea: str = "redsea"

    def as_dict(self) -> Dict[str, str]:
        return {
            "hackrf_transfer": self.hackrf_transfer,
            "hackrf_sweep": self.hackrf_sweep,
            "redsea": self.redsea,
        }

# ==========================
# Capture / decode config
# ==========================
@dataclass
class CaptureConfig:
    # station vs center
    station_freq_hz: int
    center_freq_hz: int
    offset_hz: int
    nco_freq_hz: int

    # rates / filtering
    sample_rate_hz: int
    decim: int
    fs_out: int
    cutoff_hz: int

    # RF
    lna_db: int = 40
    vga_db: int = 20
    amp: bool = False

    # capture duration
    seconds: float = 12.0

    # decoder / debug
    redsea_input: str = "full"  # "full" | "rds_bandpass"
    rds_bandpass_bw_hz: float = 4_800.0
    rds_bandpass_taps: int = 257
    settle_s: float = 0.25  # drop first N seconds (transients)

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "station_freq_hz": int(self.station_freq_hz),
            "center_freq_hz": int(self.center_freq_hz),
            "offset_hz": int(self.offset_hz),
            "nco_freq_hz": int(self.nco_freq_hz),
            "sample_rate_hz": int(self.sample_rate_hz),
            "decim": int(self.decim),
            "fs_out": int(self.fs_out),
            "cutoff_hz": int(self.cutoff_hz),
            "lna_db": int(self.lna_db),
            "vga_db": int(self.vga_db),
            "amp": bool(self.amp),
            "seconds": float(self.seconds),
            "redsea_input": str(self.redsea_input),
            "rds_bandpass_bw_hz": float(self.rds_bandpass_bw_hz),
            "rds_bandpass_taps": int(self.rds_bandpass_taps),
            "settle_s": float(self.settle_s),
        }

def make_capture_cfg(
    station_freq_mhz: float,
    *,
    offset_hz: int = 228_000,
    sample_rate_hz: int = 2_280_000,
    decim: int = 12,
    lna_db: int = 40,
    vga_db: int = 20,
    amp: bool = False,
    seconds: float = 12.0,
    redsea_input: str = "full",
) -> CaptureConfig:
    """
    Create a baseline CaptureConfig.

    Notes:
    - We intentionally tune the HackRF *offset* from the station to reduce DC spike impact.
    - Then we digitally mix (NCO) by -offset to bring the station back to baseband.
    """
    station_freq_hz = int(round(float(station_freq_mhz) * 1e6))
    center_freq_hz = station_freq_hz + int(offset_hz)
    nco_freq_hz = -int(offset_hz)

    sample_rate_hz = int(sample_rate_hz)
    decim = int(decim)
    fs_out = int(sample_rate_hz // decim)

    # We want to preserve MPX up to ~60 kHz but avoid aliasing at Nyquist.
    # Choose cutoff as ~0.47*fs_out (<= Nyquist*0.94).
    cutoff_hz = int(min(95_000, max(65_000, int(0.47 * fs_out))))

    return CaptureConfig(
        station_freq_hz=station_freq_hz,
        center_freq_hz=center_freq_hz,
        offset_hz=int(offset_hz),
        nco_freq_hz=nco_freq_hz,
        sample_rate_hz=sample_rate_hz,
        decim=decim,
        fs_out=fs_out,
        cutoff_hz=cutoff_hz,
        lna_db=int(lna_db),
        vga_db=int(vga_db),
        amp=bool(amp),
        seconds=float(seconds),
        redsea_input=str(redsea_input),
    )
