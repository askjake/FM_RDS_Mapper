# -*- coding: utf-8 -*-
"""
Scan_methods.py (v31)

Band scanning helpers:
- hackrf_sweep wrapper with forgiving parsing
- basic peak picking to suggest candidate stations
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from Defaults import FM_STEP_KHZ_DEFAULT
from Utilities import quantize_hackrf_gain, resolve_tool

@dataclass
class SweepBin:
    freq_hz: int
    power_db: float

def run_hackrf_sweep(
    hackrf_sweep_path: str,
    f_start_mhz: float,
    f_stop_mhz: float,
    *,
    bin_width_hz: int = 200_000,
    lna_db: int = 40,
    vga_db: int = 20,
    amp: bool = False,
    seconds: float = 2.5,
) -> List[SweepBin]:
    """
    Runs hackrf_sweep briefly and parses CSV-like output.

    Note: hackrf_sweep output format varies; this parser aims to be forgiving.
    """
    exe = resolve_tool(hackrf_sweep_path, ["hackrf_sweep", "hackrf_sweep.exe"])
    if not exe:
        raise RuntimeError("hackrf_sweep not found (path or PATH).")

    lna_applied, vga_applied, _ = quantize_hackrf_gain(lna_db, vga_db)

    args = [
        exe,
        "-f", f"{float(f_start_mhz)}:{float(f_stop_mhz)}",
        "-w", str(int(bin_width_hz)),
        "-l", str(int(lna_applied)),
        "-g", str(int(vga_applied)),
    ]
    if amp:
        args += ["-a", "1"]

    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    t0 = time.time()
    out_lines: List[str] = []
    try:
        while time.time() - t0 < float(seconds):
            line = p.stdout.readline()
            if not line:
                break
            out_lines.append(line.strip())
    finally:
        try:
            p.terminate()
        except Exception:
            pass
        try:
            p.wait(timeout=2)
        except Exception:
            pass

    bins: List[SweepBin] = []
    # Common hackrf_sweep line:
    # YYYY-mm-dd,HH:MM:SS,low_mhz,high_mhz,bin_width_hz,num_bins,p0,p1,...
    for line in out_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            low_mhz = float(parts[2])
            bw = int(float(parts[4]))
            pw = []
            for x in parts[6:]:
                if re.match(r"^-?\d+(\.\d+)?$", x):
                    pw.append(float(x))
            if not pw:
                continue
            low_hz = int(low_mhz * 1e6)
            for i, p_db in enumerate(pw):
                bins.append(SweepBin(freq_hz=low_hz + i * bw, power_db=float(p_db)))
        except Exception:
            continue
    return bins

def sweep_to_channel_peaks(bins: List[SweepBin], step_khz: int = FM_STEP_KHZ_DEFAULT) -> List[Dict[str, float]]:
    """
    Convert sweep bins into a channel grid and pick peak per channel.
    Returns list of dicts: {freq_hz, freq_mhz, power_db}.
    """
    if not bins:
        return []
    step_hz = int(step_khz) * 1000
    acc: Dict[int, float] = {}
    for b in bins:
        fc = int(round(int(b.freq_hz) / step_hz) * step_hz)
        acc[fc] = max(acc.get(fc, -1e9), float(b.power_db))
    out = [{"freq_hz": k, "freq_mhz": k/1e6, "power_db": v} for k, v in acc.items()]
    out.sort(key=lambda d: d["power_db"], reverse=True)
    return out