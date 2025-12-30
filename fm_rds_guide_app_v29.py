#!/usr/bin/env python3
"""
FM RDS Guide (HackRF + Streamlit) — v4

What it does
- Scan the FM broadcast band and suggest likely stations
- Tune each station, decode RDS (PS/RT/PI/PTY) and store history
- Present a "TVGuide-style" lineup + a timeline scrubber for RadioText
- Optional nightly auto-scan (in-app) + a separate CLI script for Windows Task Scheduler

Tools (must be in PATH, or place .exe beside this file)
- hackrf_transfer.exe
- (optional) hackrf_sweep.exe  (fast scan; requires FFTW DLL on Windows)
- (optional) hackrf_info.exe   (device check)
- redsea.exe                  (RDS decoder; Windows release available)

Python deps
  pip install -U streamlit pandas numpy

Run
  streamlit run fm_rds_guide_app_v3.py
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import wave
import threading
import time
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st


# ==========================
# Tool path overrides (optional)
# ==========================
APP_DIR = Path(__file__).resolve().parent
TOOLS_CONFIG_FILE = APP_DIR / "fm_tools.json"

DEFAULT_TOOL_KEYS = ["hackrf_transfer", "hackrf_sweep", "hackrf_info", "redsea"]

def load_tool_overrides() -> Dict[str, str]:
    try:
        if TOOLS_CONFIG_FILE.exists():
            d = json.loads(TOOLS_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                # Normalize: keep only known keys as strings
                out = {}
                for k in DEFAULT_TOOL_KEYS:
                    v = d.get(k)
                    if isinstance(v, str) and v.strip() and not v.strip().startswith("❌"):
                        out[k] = v.strip()
                return out
    except Exception:
        pass
    return {}

def save_tool_overrides(d: Dict[str, str]) -> None:
    payload = {k: d.get(k, "") for k in DEFAULT_TOOL_KEYS}
    TOOLS_CONFIG_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

# Loaded once; Streamlit UI can override in session_state
TOOL_OVERRIDES_DEFAULT: Dict[str, str] = load_tool_overrides()

def iso_utc_now() -> str:
    """Current time in ISO-8601 UTC (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def emit_event(q: "queue.Queue[dict]", event: dict) -> None:
    """Safely push a JSON-serializable event onto the event queue.

    Background threads MUST use this instead of Streamlit APIs.
    Adds an ISO-8601 UTC timestamp field `ts` if missing.
    """
    if not isinstance(event, dict):
        event = {"type": "debug", "msg": str(event)}
    event.setdefault("ts", iso_utc_now())
    try:
        q.put_nowait(event)
    except Exception:
        try:
            q.put(event, timeout=0.1)
        except Exception:
            # If the queue is full or interpreter shutting down, drop the event.
            pass



def tail_lines(text: str, n: int = 20) -> List[str]:
    """Return the last N lines of a possibly-multiline string."""
    if not text:
        return []
    # Normalize newlines then split. Keep even if last line empty? We drop trailing empty.
    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    while lines and lines[-1] == "":
        lines.pop()
    return lines[-n:]

def get_tool_overrides() -> Dict[str, str]:
    # Prefer Streamlit session state if available
    try:
        if "tool_overrides" in st.session_state and isinstance(st.session_state.tool_overrides, dict):
            return st.session_state.tool_overrides
    except Exception:
        pass
    return TOOL_OVERRIDES_DEFAULT

def resolve_tool(tool_key: str, exe_names: List[str], explicit_path: Optional[str] = None) -> Optional[str]:
    # Allow a thread-safe explicit override (e.g. passed in via CaptureConfig).
    if explicit_path:
        p = Path(str(explicit_path))
        if p.exists():
            return str(p)
    try:
        overrides = get_tool_overrides()
    except Exception:
        overrides = dict(TOOL_OVERRIDES_DEFAULT)
    override = overrides.get(tool_key)
    if override:
        p = Path(override)
        if p.exists():
            return str(p)
    # PATH / local file fallback
    for name in exe_names:
        p = which_or_none(name)
        if p:
            return p
    return None



def resolve_tools(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve all external tool paths we rely on.

    This is thread-safe: it does NOT require Streamlit context, and it will fall back to:
      1) explicit overrides passed in,
      2) TOOL_OVERRIDES_DEFAULT loaded from disk at import time,
      3) system PATH lookup.
    """
    # Start with defaults loaded at import time (may come from tool_overrides.json).
    merged: dict[str, str] = dict(TOOL_OVERRIDES_DEFAULT) if isinstance(TOOL_OVERRIDES_DEFAULT, dict) else {}
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, str) and v.strip():
                merged[k] = v.strip()

    # Try to incorporate session overrides if we happen to be running in the Streamlit main thread.
    try:
        if "tool_overrides" in st.session_state and isinstance(st.session_state.tool_overrides, dict):
            for k, v in st.session_state.tool_overrides.items():
                if isinstance(v, str) and v.strip():
                    merged[k] = v.strip()
    except Exception:
        pass

    tools: dict[str, str] = {}
    tools["hackrf_transfer"] = resolve_tool("hackrf_transfer", ["hackrf_transfer.exe", "hackrf_transfer"], merged.get("hackrf_transfer"))
    tools["hackrf_sweep"] = resolve_tool("hackrf_sweep", ["hackrf_sweep.exe", "hackrf_sweep"], merged.get("hackrf_sweep"))
    tools["redsea"] = resolve_tool("redsea", ["redsea.exe", "redsea"], merged.get("redsea"))
    return tools


def _find_windows_dll(dll_name: str, exe_path: str) -> Optional[str]:
    """
    Best-effort check to prevent the Windows "System Error" popup when a DLL dependency is missing.
    We search: exe directory, PATH entries, current working directory.
    """
    if os.name != "nt":
        return None
    try:
        exe_dir = Path(exe_path).resolve().parent
        p = exe_dir / dll_name
        if p.exists():
            return str(p)
    except Exception:
        pass

    for d in os.environ.get("PATH", "").split(os.pathsep):
        d = (d or "").strip().strip('"')
        if not d:
            continue
        try:
            p = Path(d) / dll_name
            if p.exists():
                return str(p)
        except Exception:
            continue

    try:
        p = Path.cwd() / dll_name
        if p.exists():
            return str(p)
    except Exception:
        pass
    return None


def hackrf_sweep_dependency_status(exe_path: str) -> Tuple[bool, str]:
    """
    hackrf_sweep on Windows needs FFTW: libfftw3f-3.dll.
    Returns (ok, detail). If ok==True, detail is the DLL path found (or empty on non-Windows).
    """
    if not exe_path:
        return False, "hackrf_sweep.exe not found"
    if os.name != "nt":
        return True, ""
    dll = _find_windows_dll("libfftw3f-3.dll", exe_path)
    if dll:
        return True, dll
    return False, "libfftw3f-3.dll not found"

# ==========================
# Defaults
# ==========================
FM_MIN_MHZ_DEFAULT = 88.0
FM_MAX_MHZ_DEFAULT = 108.0
FM_STEP_KHZ_DEFAULT = 200

# fast scan (hackrf_sweep)
SWEEP_BIN_WIDTH_HZ_DEFAULT = 200_000
SWEEP_LNA_DB_DEFAULT = 56
SWEEP_VGA_DB_DEFAULT = 20
SWEEP_AMP_DEFAULT = False

# fallback scan (step tune + power estimate)
STEP_SCAN_DWELL_MS_DEFAULT = 60

# capture + decode
CAPTURE_FS_HZ_DEFAULT = 2_052_000   # decim by 12 -> 171 kHz MPX (redsea-friendly)
CAPTURE_DECIM_DEFAULT = 12
CAPTURE_SECONDS_DEFAULT = 6.0

TRANSFER_LNA_DB_DEFAULT = 56
TRANSFER_VGA_DB_DEFAULT = 20
TRANSFER_AMP_DEFAULT = False

PEAK_THRESHOLD_DB_DEFAULT = 10.0
MAX_STATIONS_DEFAULT = 30

DB_PATH_DEFAULT = "fm_rds_guide.sqlite3"

# HackRF gain constraints:
# - LNA/IF gain (-l): 0..40 dB in 8 dB steps
# - VGA/baseband gain (-g): 0..62 dB in 2 dB steps
def _quantize_step(value: int, step: int) -> int:
    return int(round(int(value) / int(step)) * int(step))

def clamp_hackrf_lna_db(lna_db: int) -> int:
    return max(0, min(40, _quantize_step(lna_db, 8)))

def clamp_hackrf_vga_db(vga_db: int) -> int:
    return max(0, min(62, _quantize_step(vga_db, 2)))



def iq_from_bytes_interleaved_int8(raw: bytes, carry: bytes = b"") -> tuple[np.ndarray, bytes]:
    """Convert interleaved int8 IQIQ... bytes to complex64 samples in roughly [-1, 1].

    On Windows/pipe streams, reads can return an odd number of bytes (splitting an I/Q pair).
    We keep a 1-byte carry so we never misalign I and Q.
    """
    if carry:
        raw = carry + raw
    if not raw:
        return np.empty((0,), dtype=np.complex64), b""
    if len(raw) < 2:
        # Not enough for one IQ pair; keep as carry
        return np.empty((0,), dtype=np.complex64), raw

    if (len(raw) % 2) == 1:
        carry = raw[-1:]
        raw = raw[:-1]
    else:
        carry = b""

    x = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    i = x[0::2]
    q = x[1::2]
    n = min(i.shape[0], q.shape[0])
    if n <= 0:
        return np.empty((0,), dtype=np.complex64), carry

    iq = (i[:n] + 1j * q[:n]) / 128.0
    return iq.astype(np.complex64), carry

# Nightly scan default (local time)
NIGHTLY_HOUR_DEFAULT = 2
NIGHTLY_MINUTE_DEFAULT = 15


# ==========================
# Utilities
# ==========================

def which_or_none(name: str) -> Optional[str]:
    return shutil.which(name) or (os.path.abspath(name) if os.path.exists(name) else None)

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def now_local() -> datetime:
    return datetime.now().astimezone()

def wav_header_pcm16_mono(sample_rate: int, data_bytes: int = 0) -> bytes:
    """Minimal WAV header for PCM16 mono. data_bytes=0 works for many decoders."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    riff_size = 36 + data_bytes
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size & 0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_bytes & 0xFFFFFFFF,
    )

def robust_json_loads(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        return None

def fmt_freq(freq_hz: int) -> str:
    return f"{freq_hz/1e6:.1f} MHz"


# ==========================
# SQLite storage
# ==========================

def db_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def db_init(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS stations (
        freq_hz INTEGER PRIMARY KEY,
        first_seen TEXT,
        last_seen TEXT,
        power_db REAL,
        pi TEXT,
        ps TEXT,
        pty INTEGER,
        rt TEXT
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS rds_groups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        freq_hz INTEGER,
        pi TEXT,
        ps TEXT,
        pty INTEGER,
        rt TEXT,
        raw_json TEXT
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)
    conn.commit()

def db_set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("""
      INSERT INTO settings(key,value) VALUES(?,?)
      ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, value))
    conn.commit()

def db_get_setting(conn: sqlite3.Connection, key: str, default: str) -> str:
    cur = conn.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else default

def db_upsert_station(conn: sqlite3.Connection, freq_hz: int, power_db: float,
                     pi: Optional[str], ps: Optional[str], pty: Optional[int], rt: Optional[str]) -> None:
    ts = now_utc_iso()
    cur = conn.cursor()
    cur.execute("SELECT first_seen FROM stations WHERE freq_hz=?", (freq_hz,))
    row = cur.fetchone()
    first_seen = row[0] if row else ts
    cur.execute("""
        INSERT INTO stations (freq_hz, first_seen, last_seen, power_db, pi, ps, pty, rt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(freq_hz) DO UPDATE SET
          last_seen=excluded.last_seen,
          power_db=excluded.power_db,
          pi=COALESCE(excluded.pi, stations.pi),
          ps=COALESCE(excluded.ps, stations.ps),
          pty=COALESCE(excluded.pty, stations.pty),
          rt=COALESCE(excluded.rt, stations.rt)
    """, (freq_hz, first_seen, ts, float(power_db), pi, ps, pty, rt))
    conn.commit()

def db_insert_group(conn: sqlite3.Connection, freq_hz: int, group: dict) -> None:
    ts = now_utc_iso()
    conn.execute("""
        INSERT INTO rds_groups (ts, freq_hz, pi, ps, pty, rt, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        ts, freq_hz, group.get("pi"), group.get("ps"), group.get("pty"),
        group.get("rt"), json.dumps(group, ensure_ascii=False)
    ))
    conn.commit()

def db_load_stations(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT freq_hz, first_seen, last_seen, power_db, pi, ps, pty, rt
        FROM stations
        ORDER BY freq_hz ASC
    """, conn)
    if df.empty:
        return df
    df["freq_mhz"] = df["freq_hz"] / 1e6
    return df

def db_load_history(conn: sqlite3.Connection, freq_hz: int, limit: int = 1000) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT ts, pi, ps, pty, rt, raw_json
        FROM rds_groups
        WHERE freq_hz=?
        ORDER BY id DESC
        LIMIT ?
    """, conn, params=(freq_hz, limit))

def lineup_by_pi(stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    "Channel lineup" learning: group freqs by PI (repeaters/translators).
    If PI missing, keep station separate (can extend later with fuzzy PS).
    """
    if stations_df.empty:
        return pd.DataFrame(columns=["pi","ps","freqs_mhz","last_seen","best_power_db","rt_sample","n_freqs"])

    df = stations_df.copy()
    df["freq_mhz"] = df["freq_hz"] / 1e6
    df["pi"] = df["pi"].fillna("")
    df["ps"] = df["ps"].fillna("")
    df["rt"] = df["rt"].fillna("")

    groups = []
    for pi, g in df.groupby("pi", dropna=False):
        # PI='' means unknown; treat each frequency as its own "lineup item"
        if not pi:
            for _, r in g.iterrows():
                groups.append({
                    "pi": "",
                    "ps": r["ps"],
                    "freqs_mhz": f"{r['freq_mhz']:.1f}",
                    "last_seen": r["last_seen"],
                    "best_power_db": r["power_db"],
                    "rt_sample": (r["rt"] or "")[:80],
                    "n_freqs": 1,
                })
            continue

        freqs = sorted([float(x) for x in g["freq_mhz"].tolist()])
        best = g.sort_values("power_db", ascending=False).iloc[0]
        # "canonical" PS = most recent non-empty
        canon_ps = ""
        for _, r in g.sort_values("last_seen", ascending=False).iterrows():
            if isinstance(r["ps"], str) and r["ps"].strip():
                canon_ps = r["ps"].strip()
                break
        if not canon_ps:
            canon_ps = best["ps"]

        rt_sample = ""
        for _, r in g.sort_values("last_seen", ascending=False).iterrows():
            if isinstance(r["rt"], str) and r["rt"].strip():
                rt_sample = r["rt"].strip()
                break

        groups.append({
            "pi": pi,
            "ps": canon_ps,
            "freqs_mhz": ", ".join(f"{f:.1f}" for f in freqs),
            "last_seen": g["last_seen"].max(),
            "best_power_db": float(best["power_db"]),
            "rt_sample": rt_sample[:80],
            "n_freqs": len(freqs),
        })
    out = pd.DataFrame(groups)
    # Sort like a TV lineup: by PS then by first freq
    def first_freq(s: str) -> float:
        try:
            return float(s.split(",")[0].strip())
        except Exception:
            return 999.0
    out["_first"] = out["freqs_mhz"].apply(first_freq)
    out = out.sort_values(["ps","_first"]).drop(columns=["_first"])
    return out


# ==========================
# Scan methods
# ==========================

def parse_hackrf_sweep_csv(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        # date/time sanity
        if "-" not in parts[0] or ":" not in parts[1]:
            continue
        try:
            hz_low = int(float(parts[2]))
            bin_w = float(parts[4])
            num_samples = int(float(parts[5]))
            db_bins = [float(x) for x in parts[6:]]
        except Exception:
            continue
        for i, p_db in enumerate(db_bins):
            hz = hz_low + int((i + 0.5) * bin_w)
            rows.append((hz, float(p_db), float(bin_w), int(num_samples)))
    if not rows:
        return pd.DataFrame(columns=["freq_hz","power_db","bin_w_hz","num_samples"])
    return pd.DataFrame(rows, columns=["freq_hz","power_db","bin_w_hz","num_samples"])

def run_hackrf_sweep(freq_min_mhz: float, freq_max_mhz: float,
                    bin_width_hz: int, lna_db: int, vga_db: int, amp: bool) -> pd.DataFrame:
    exe = resolve_tool("hackrf_sweep", ["hackrf_sweep", "hackrf_sweep.exe"])
    if not exe:
        raise RuntimeError("hackrf_sweep not found in PATH.")
    ok, detail = hackrf_sweep_dependency_status(exe)
    if not ok:
        raise RuntimeError(
            "hackrf_sweep is present, but can't run on this system because a dependency is missing.\n\n"
            f"  {detail}\n\n"
            "Fix: copy libfftw3f-3.dll next to hackrf_sweep.exe (e.g., C:\\HackRF\\bin), "
            "or install a HackRF tools bundle that includes it.\n"
            "Workaround: in the app, choose 'Step scan (no hackrf_sweep)'."
        )
    # NOTE: Some Windows builds of hackrf_sweep only accept integer MHz ranges (e.g. 88:108)
    # and reject floats like 88.0:108.0. We'll try floats first, then retry with ints if needed.
    f_arg_float = f"{float(freq_min_mhz)}:{float(freq_max_mhz)}"
    fmin_i = int(math.floor(float(freq_min_mhz)))
    fmax_i = int(math.ceil(float(freq_max_mhz)))
    f_arg_int = f"{fmin_i}:{fmax_i}"

    # Quantize/clamp gains to HackRF supported ranges
    lna_req = int(lna_db)
    vga_req = int(vga_db)
    lna_db = clamp_hackrf_lna_db(lna_req)
    vga_db = clamp_hackrf_vga_db(vga_req)
    if (lna_db != lna_req) or (vga_db != vga_req):
        print(f"ℹ️  hackrf_sweep gain adjusted: requested LNA={lna_req} VGA={vga_req} -> applied LNA={lna_db} VGA={vga_db}")

    def run_once(f_arg: str) -> subprocess.CompletedProcess:
        cmd = [
            exe,
            "-f", f_arg,
            "-w", str(int(bin_width_hz)),
            "-l", str(int(lna_db)),
            "-g", str(int(vga_db)),
            "-a", "1" if amp else "0",
            "-1",
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=40)

    p = run_once(f_arg_float)
    if p.returncode != 0:
        combined0 = (p.stderr or "") + "\n" + (p.stdout or "")
        if ("invalid parameter" in combined0.lower() or "argument error" in combined0.lower()) and "-f" in combined0:
            p = run_once(f_arg_int)
    if p.returncode != 0:
        combined = (p.stderr or "") + "\n" + (p.stdout or "")
        if "libfftw3f-3.dll" in combined.lower():
            raise RuntimeError(
                "hackrf_sweep failed because libfftw3f-3.dll is missing.\n\n"
                "Fix: put libfftw3f-3.dll next to hackrf_sweep.exe (C:\\HackRF\\bin), "
                "or reinstall a complete HackRF tools bundle that includes it.\n"
                "Workaround: switch Scan Method to 'Step scan (no hackrf_sweep)'."
            )
        raise RuntimeError(f"hackrf_sweep failed:\n{combined.strip()}")
    return parse_hackrf_sweep_csv(p.stdout)

def detect_fm_station_peaks(sweep_df: pd.DataFrame,
                            fm_step_khz: int,
                            threshold_db_above_median: float,
                            max_stations: int) -> pd.DataFrame:
    if sweep_df.empty:
        return pd.DataFrame(columns=["freq_hz","freq_mhz","power_db"])
    freqs = sweep_df["freq_hz"].to_numpy()
    pwr = sweep_df["power_db"].to_numpy()
    noise = float(np.median(pwr))
    step_hz = int(fm_step_khz * 1000)
    fmin = int(freqs.min())
    fmax = int(freqs.max())
    grid_start = (fmin // step_hz) * step_hz
    grid = np.arange(grid_start, fmax + step_hz, step_hz, dtype=np.int64)
    half = step_hz // 2

    candidates = []
    for fc in grid:
        mask = (freqs >= (fc - half)) & (freqs < (fc + half))
        if not np.any(mask):
            continue
        candidates.append((int(fc), float(np.max(pwr[mask]))))
    if not candidates:
        return pd.DataFrame(columns=["freq_hz","freq_mhz","power_db"])

    cand_df = pd.DataFrame(candidates, columns=["freq_hz","power_db"]).sort_values("power_db", ascending=False)
    cand_df = cand_df[cand_df["power_db"] >= (noise + threshold_db_above_median)]

    kept = []
    used = set()
    for _, r in cand_df.iterrows():
        fc = int(r["freq_hz"])
        if fc in used:
            continue
        kept.append((fc, float(r["power_db"])))
        for n in (fc - step_hz, fc, fc + step_hz):
            used.add(n)
        if len(kept) >= max_stations:
            break

    out = pd.DataFrame(kept, columns=["freq_hz","power_db"]).sort_values("freq_hz")
    out["freq_mhz"] = out["freq_hz"] / 1e6
    return out[["freq_hz","freq_mhz","power_db"]]

@dataclass
class StepScanCfg:
    freq_hz: int
    sample_rate_hz: int
    vga_db: int
    lna_db: int
    amp: bool
    dwell_ms: int

class HackRFTransfer:
    """
    Wrap hackrf_transfer streaming IQ to stdout. Drains stderr so it can't deadlock.
    Output: interleaved signed 8-bit IQ.
    """
    def __init__(self, hackrf_transfer_path: Optional[str] = None):
        self.hackrf_transfer_path = hackrf_transfer_path
        self.proc: Optional[subprocess.Popen] = None
        self._stderr_tail: List[str] = []
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._carry: bytes = b""
        self.sample_rate_hz: Optional[int] = None
        self._default_read_bytes: int = 262144

    def start(self, freq_hz: Optional[int] = None, sample_rate_hz: int = 0, vga_db: int = 0, lna_db: int = 0, amp: bool = False, *, center_freq_hz: Optional[int] = None, amp_enable: Optional[bool] = None):
        # Accept legacy keyword aliases
        if freq_hz is None and center_freq_hz is not None:
            freq_hz = int(center_freq_hz)
        if amp_enable is not None:
            amp = bool(amp_enable)
        if freq_hz is None:
            raise ValueError("freq_hz (or center_freq_hz) is required")

        vga_db_req = int(vga_db)
        lna_db_req = int(lna_db)
        vga_db = clamp_hackrf_vga_db(vga_db_req)
        lna_db = clamp_hackrf_lna_db(lna_db_req)

        # Remember sample rate for auto-sized reads
        try:
            self.sample_rate_hz = int(sample_rate_hz)
            # Default to ~0.25s chunks (bytes = Fs * 2 bytes/complex-sample * seconds)
            self._default_read_bytes = max(16384, int(self.sample_rate_hz * 2 * 0.25))
            # Ensure even number of bytes (I/Q interleaved)
            self._default_read_bytes -= (self._default_read_bytes % 2)
        except Exception:
            self.sample_rate_hz = None
            self._default_read_bytes = 262144

        exe = resolve_tool("hackrf_transfer", ["hackrf_transfer", "hackrf_transfer.exe"], explicit_path=self.hackrf_transfer_path)
        if not exe:
            raise RuntimeError("hackrf_transfer not found in PATH.")

        cmd = [
            exe, "-r", "-",
            "-f", str(int(freq_hz)),
            "-s", str(int(sample_rate_hz)),
            "-g", str(int(vga_db)),
            "-l", str(int(lna_db)),
            "-a", "1" if amp else "0",
        ]

        self._stop_evt.clear()
        self._carry = b""  # reset IQ alignment
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

        def drain():
            assert self.proc and self.proc.stderr
            for raw in iter(self.proc.stderr.readline, b""):
                if self._stop_evt.is_set():
                    break
                s = raw.decode("utf-8", errors="replace").rstrip()
                with self._lock:
                    self._stderr_tail.append(s)
                    self._stderr_tail = self._stderr_tail[-80:]
        threading.Thread(target=drain, daemon=True).start()

        # Return useful info for debug (not required by callers)
        return {
            "cmd": cmd,
            "requested": {"vga_db": vga_db_req, "lna_db": lna_db_req, "amp": bool(amp)},
            "applied": {"vga_db": int(vga_db), "lna_db": int(lna_db), "amp": bool(amp)},
        }

    def read(self, nbytes: int) -> bytes:
        if not self.proc or not self.proc.stdout:
            return b""
        return self.proc.stdout.read(nbytes)


    def read_iq(self, nbytes: Optional[int] = None) -> np.ndarray:
        """Read raw bytes and return aligned complex IQ samples.

        If nbytes is None, uses an auto-sized chunk derived from the last configured sample rate.
        """
        if nbytes is None:
            nbytes = int(getattr(self, "_default_read_bytes", 262144))
        try:
            nbytes = int(nbytes)
        except Exception:
            nbytes = 262144
        if nbytes <= 0:
            return np.empty((0,), dtype=np.complex64)
        raw = self.read(nbytes)
        if not raw:
            return np.empty((0,), dtype=np.complex64)
        iq, self._carry = iq_from_bytes_interleaved_int8(raw, self._carry)
        return iq

    def stop(self):
        self._stop_evt.set()
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
        self._carry = b""

    def stderr_tail(self) -> str:
        with self._lock:
            return "\n".join(self._stderr_tail)

def step_scan_power(fmin_mhz: float, fmax_mhz: float, step_khz: int,
                    sample_rate_hz: int, dwell_ms: int, vga_db: int, lna_db: int, amp: bool,
                    max_stations: int, threshold_db_above_median: float) -> pd.DataFrame:
    """
    No-hackrf_sweep fallback:
    For each FM channel, tune with hackrf_transfer, read a short IQ burst, estimate power.
    """
    step_hz = int(step_khz * 1000)
    fmin = int(round(fmin_mhz * 1e6))
    fmax = int(round(fmax_mhz * 1e6))

    grid_start = (fmin // step_hz) * step_hz
    freqs = list(range(grid_start, fmax + 1, step_hz))

    # bytes to read: int8 IQ => 2 bytes per complex
    n_complex = int(sample_rate_hz * (dwell_ms / 1000.0))
    n_bytes = max(4096, n_complex * 2)

    powers = []
    for f in freqs:
        transfer = HackRFTransfer()
        transfer.start(f, sample_rate_hz, vga_db, lna_db, amp)
        raw = transfer.read(n_bytes)
        transfer.stop()
        if not raw or len(raw) < 4096:
            powers.append((f, -200.0))
            continue
        iq, _ = iq_from_bytes_interleaved_int8(raw, b"")
        if iq.size == 0:
            powers.append((f, -200.0))
            continue
        p = np.abs(iq) ** 2
        p95 = float(np.percentile(p, 95))
        db = 10.0 * math.log10(p95 + 1e-12)
        powers.append((f, db))

    df = pd.DataFrame(powers, columns=["freq_hz","power_db"])
    noise = float(np.median(df["power_db"]))
    cand = df[df["power_db"] >= (noise + threshold_db_above_median)].sort_values("power_db", ascending=False)

    kept = []
    used = set()
    for _, r in cand.iterrows():
        fc = int(r["freq_hz"])
        if fc in used:
            continue
        kept.append((fc, float(r["power_db"])))
        for n in (fc - step_hz, fc, fc + step_hz):
            used.add(n)
        if len(kept) >= max_stations:
            break

    out = pd.DataFrame(kept, columns=["freq_hz","power_db"]).sort_values("freq_hz")
    out["freq_mhz"] = out["freq_hz"] / 1e6
    return out[["freq_hz","freq_mhz","power_db"]]


# ==========================
# FM demod + RDS decode
# ==========================

def fm_quadrature_demod(iq: np.ndarray) -> np.ndarray:
    prod = iq[1:] * np.conj(iq[:-1])
    return np.angle(prod).astype(np.float32)



class NCO:
    """Simple NCO for offset tuning / digital mixing."""
    def __init__(self, fs: float, freq_hz: float):
        self.fs = float(fs)
        self.freq = float(freq_hz)
        self.phase = 0.0

    def step(self, n: int) -> np.ndarray:
        """Return the next `n` oscillator samples as exp(-j*phase).

        We keep this as a separate method (in addition to `mix_down`) because
        several earlier revisions of this app used `nco.step(n)` and later
        revisions refactored to `nco.mix_down(x)`. Having both prevents
        "missing attribute" crashes when capture logic evolves.
        """
        n = int(n)
        if n <= 0:
            return np.empty((0,), dtype=np.complex64)
        if self.freq == 0.0:
            return np.ones((n,), dtype=np.complex64)

        idx = np.arange(n, dtype=np.float32)
        w = (2.0 * np.pi * self.freq / self.fs)
        ph = self.phase + w * idx
        osc = np.exp(-1j * ph).astype(np.complex64)
        # Advance phase accumulator
        self.phase = float((ph[-1] + w) % (2.0 * np.pi))
        return osc

    def mix_down(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.complex64, copy=False)
        osc = self.step(x.size)
        return (x.astype(np.complex64, copy=False) * osc).astype(np.complex64)



class MovingRMS:
    """Exponential moving RMS estimator.

    Used as a gentle AGC reference on the demodulated MPX signal.
    """
    def __init__(self, alpha: float = 0.05, eps: float = 1e-12):
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._ms = float(eps)

    def update(self, x: np.ndarray) -> float:
        if x is None:
            return float(math.sqrt(self._ms))
        # x is expected to be float array
        try:
            ms = float(np.mean(np.square(x, dtype=np.float64)))
        except Exception:
            ms = 0.0
        a = self.alpha
        self._ms = (1.0 - a) * self._ms + a * max(self.eps, ms)
        return float(math.sqrt(self._ms))

    @property
    def rms(self) -> float:
        return float(math.sqrt(self._ms))
class FIRDecimator:
    def __init__(self, decim: int, fs_in: float, cutoff_hz: float, taps: int = 161):
        self.decim = int(decim)
        self.taps = self._design_lowpass(fs_in, cutoff_hz, taps).astype(np.float32)
        self.zi = np.zeros(len(self.taps) - 1, dtype=np.float32)

    @staticmethod
    def _design_lowpass(fs: float, cutoff: float, taps: int) -> np.ndarray:
        fc = cutoff / fs
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(2 * fc * n) * (2 * fc)
        h *= np.hamming(taps)
        h /= np.sum(h)
        return h

    def process(self, x: np.ndarray) -> np.ndarray:
        x_full = np.concatenate([self.zi, x])
        y = np.convolve(x_full, self.taps, mode="valid").astype(np.float32)
        self.zi = x_full[-(len(self.taps) - 1):].copy()
        return y[::self.decim]


class ComplexFIRDecimator:
    """Low-pass FIR + decimation for complex IQ."""
    def __init__(self, decim: int, fs_in: float, cutoff_hz: float, taps: int = 161):
        self.decim = int(decim)
        self.taps = FIRDecimator._design_lowpass(fs_in, cutoff_hz, taps).astype(np.float32)
        self.zi = np.zeros(len(self.taps) - 1, dtype=np.complex64)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.complex64)
        x = x.astype(np.complex64, copy=False)
        x_full = np.concatenate([self.zi, x]) if self.zi.size else x
        y = np.convolve(x_full, self.taps, mode="valid").astype(np.complex64)
        self.zi = x_full[-(len(self.taps) - 1):].copy()
        return y[::self.decim]


class RedseaDecoder:
    """
    Wrapper around `redsea` for offline decoding.

    Why offline? On Windows, many CLI tools block-buffer stdout when it's a pipe,
    so "live" line-by-line JSON can appear as "no groups" even when decode works.
    This class favors correctness:
      1) capture MPX to a WAV (mono, 16-bit)
      2) run redsea on the file
      3) parse JSON lines from stdout
    """

    def __init__(self, redsea_exe: str):
        self.redsea_exe = redsea_exe
        self.last_cmd: list[str] = []
        self.last_returncode: int | None = None
        self.last_stdout: str = ""
        self.last_stderr: str = ""

    @staticmethod
    def _normalize_group(obj: dict) -> dict:
        """
        Normalize redsea JSON objects into a common shape used by our DB/UI.
        We try several possible key spellings to be resilient across redsea builds.
        """
        def pick(*keys):
            for k in keys:
                if k in obj and obj[k] not in (None, "", []):
                    return obj[k]
            return None

        pi = pick("pi", "PI")
        # If PI is an int, represent it as 4 hex digits (common display format)
        if isinstance(pi, int):
            pi = f"{pi:04X}"
        elif isinstance(pi, str):
            pi = pi.strip() or None

        ps = pick("ps", "PS", "program_service", "program_service_name", "station", "callsign")
        if isinstance(ps, str):
            ps = ps.strip() or None

        rt = pick("rt", "RT", "radiotext", "radio_text", "text")
        if isinstance(rt, str):
            rt = rt.strip() or None

        pty = pick("pty", "PTY", "program_type")
        if isinstance(pty, str):
            try:
                pty = int(pty)
            except Exception:
                pass

        out = {
            "pi": pi,
            "ps": ps,
            "pty": pty,
            "rt": rt,
            "raw": obj,
        }
        return out

    def decode_wav(
        self,
        wav_path: str,
        *,
        rbds: bool = True,
        show_partial: bool = True,
        time_from_start: bool = True,
        timeout_s: float = 20.0,
    ) -> tuple[list[dict], list[str], list[str], int, int]:
        """
        Returns:
          groups: normalized group dicts
          stdout_lines, stderr_lines: full line lists
          returncode: process returncode
          parse_errors: JSON lines that failed to parse
        """
        cmd = [self.redsea_exe, "-f", wav_path, "-o", "json"]
        if rbds:
            cmd.append("-u")
        if show_partial:
            cmd.append("-p")
        if time_from_start:
            cmd.append("--time-from-start")

        self.last_cmd = cmd

        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            # Best effort: capture any partial output
            self.last_returncode = -1
            self.last_stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
            self.last_stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
            return [], (self.last_stdout or "").splitlines(), (self.last_stderr or "").splitlines(), -1, 0

        self.last_returncode = p.returncode
        self.last_stdout = p.stdout or ""
        self.last_stderr = p.stderr or ""

        stdout_lines = self.last_stdout.splitlines()
        stderr_lines = self.last_stderr.splitlines()

        groups: list[dict] = []
        parse_errors = 0
        for line in stdout_lines:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                parse_errors += 1
                continue
            if isinstance(obj, dict):
                groups.append(self._normalize_group(obj))

        return groups, stdout_lines, stderr_lines, p.returncode, parse_errors


@dataclass
class CaptureConfig:
    station_freq_hz: int
    center_freq_hz: int
    offset_hz: int
    sample_rate_hz: int
    decim: int
    cutoff_hz: int
    lna_db: int
    vga_db: int
    amp: bool
    seconds: float
    tool_overrides: dict[str, str] = field(default_factory=dict)

    # Backwards-compat attribute aliases (older code versions used these names).
    @property
    def amp_enable(self) -> bool:
        return bool(self.amp)

    @property
    def seconds_per_station(self) -> float:
        return float(self.seconds)

    @property
    def tool_paths_overrides(self) -> dict[str, str]:
        return dict(self.tool_overrides) if isinstance(self.tool_overrides, dict) else {}


def decode_rds_for_station(cfg: CaptureConfig, event_q: "queue.Queue[dict]", db_path: str, power_db: float | None) -> tuple[bool, int]:
    """
    Capture IQ with hackrf_transfer, demod WBFM to MPX, then decode RDS/RBDS with redsea.

    This function runs in a background thread. It MUST NOT call Streamlit APIs.
    It communicates exclusively via `event_q`.
    
    Returns:
        (success: bool, num_groups: int)
        - success=True if at least one RDS group was decoded
        - num_groups: total number of groups decoded
    """
    station_freq_hz = int(cfg.center_freq_hz - cfg.offset_hz)
    center_freq_hz = int(cfg.center_freq_hz)
    offset_hz = int(cfg.offset_hz)
    nco_freq_hz = -offset_hz  # we tune above and mix back down

    fs_in = int(cfg.sample_rate_hz)
    decim = int(cfg.decim)
    fs_out = int(fs_in // decim)

    # WBFM channelization: keep enough bandwidth for the full composite (pilot 19k, stereo 38k, RDS 57k)
    # Nyquist is fs_out/2; keep a little margin to avoid aliasing.
    # Channel LPF cutoff is expressed in Hz relative to the *input* IQ sample rate.
    # Prefer the value carried in cfg (so the UI/monitor thread can control it),
    # but also clamp it to something safe for the chosen decimation.
    cutoff_hz = int(getattr(cfg, "cutoff_hz", min(160_000, 0.49 * fs_out)))
    cutoff_hz = int(min(cutoff_hz, 0.49 * fs_out))

    # MPX scaling: quadrature demod returns radians/sample ~= 2π * f_dev / fs_out.
    # Scale so ±75 kHz deviation maps to roughly ±1.0 MPX units.
    FM_DEV_HZ = 75_000.0
    demod_scale = float(fs_out) / (2.0 * math.pi * FM_DEV_HZ)
    mpx_dc = 0.0
    prev_iq: complex | None = None
    mpx_dc_alpha = 0.001
    mpx_clip_samples = 0
    mpx_total_samples = 0

    # Diagnostic cadence
    diag_interval_s = 1.0

    # Early exit tracking
    min_groups_to_proceed = 1  # Move on after decoding at least 1 group
    check_interval_s = 1.0  # Check every second if we can exit early
    decoded_groups_so_far = 0
    last_check = 0.0

    emit_event(event_q, {
        "type": "debug",
        "stage": "station_start",
        "station_freq_hz": station_freq_hz,
        "center_freq_hz": center_freq_hz,
        "offset_hz": offset_hz,
        "nco_freq_hz": nco_freq_hz,
        "sample_rate_hz": fs_in,
        "decim": decim,
        "fs_out": fs_out,
        "cutoff_hz": cutoff_hz,
        "vga_db": int(cfg.vga_db),
        "lna_db": int(cfg.lna_db),
        "amp": bool(cfg.amp_enable),
        "seconds": float(cfg.seconds_per_station),
        "ts": iso_utc_now(),
    })

    tools = resolve_tools(cfg.tool_paths_overrides)
    transfer = HackRFTransfer(tools["hackrf_transfer"])

    # Start capture (tune to the *center*; we NCO the offset back to baseband)
    t0 = time.time()
    try:
        transfer.start(
            freq_hz=int(cfg.center_freq_hz),
            sample_rate_hz=int(fs_in),
            vga_db=int(cfg.vga_db),
            lna_db=int(cfg.lna_db),
            amp=bool(cfg.amp),
        )
    except Exception as e:
        emit_event(event_q, {"type": "error", "msg": f"hackrf_transfer failed to start: {e}", "ts": iso_utc_now()})
        return (False, 0)

    # DSP state
    chan = ComplexFIRDecimator(fs_in=fs_in, decim=decim, cutoff_hz=cutoff_hz)
    nco = NCO(fs=fs_in, freq_hz=float(nco_freq_hz))
    agc = MovingRMS(alpha=0.02)  # slow-ish RMS tracker on demod output

    # Capture MPX as PCM16; later write to WAV and run redsea offline for reliable stdout parsing.
    pcm_chunks: list[np.ndarray] = []
    total_iq_samples = 0
    total_pcm_samples = 0
    clip_pct_latest = 0.0
    iq_rms_latest = 0.0
    iq_dc_i_latest = 0.0
    iq_dc_q_latest = 0.0
    pilot_snr_db_latest: float | None = None
    rds_snr_db_latest: float | None = None
    agc_rms_latest = 0.0

    diag_next = t0
    t_end = t0 + float(cfg.seconds_per_station)

    # Consume chunks until time elapsed OR we've decoded enough
    try:
        while time.time() < t_end:
            iq = transfer.read_iq()
            if iq is None or iq.size == 0:
                break
            total_iq_samples += int(iq.size)

            # Basic IQ stats
            i = np.real(iq).astype(np.float32, copy=False)
            q = np.imag(iq).astype(np.float32, copy=False)
            iq_rms = float(np.sqrt(np.mean(i * i + q * q)))
            clip_pct = float(np.mean((np.abs(i) >= 0.99) | (np.abs(q) >= 0.99)) * 100.0)
            iq_dc_i = float(np.mean(i))
            iq_dc_q = float(np.mean(q))

            # Mix offset -> baseband then channelize/decimate
            bb = iq * nco.step(iq.size)
            bb_ds = chan.process(bb)

            # FM demod => MPX (composite baseband)
            # FM quadrature demod with block-to-block continuity (output length == len(bb_ds)).
            if prev_iq is None:
                prev_iq = complex(bb_ds[0])
            shifted = np.empty_like(bb_ds)
            shifted[0] = prev_iq
            shifted[1:] = bb_ds[:-1]
            phase_diff = np.angle(bb_ds * np.conj(shifted))
            prev_iq = complex(bb_ds[-1])
            mpx = phase_diff * demod_scale
            # Persistent DC blocker: per-block mean subtraction can smear low-freq content and disturb subcarriers
            mpx_dc = (1.0 - mpx_dc_alpha) * mpx_dc + mpx_dc_alpha * float(np.mean(mpx))
            mpx = mpx - mpx_dc
            mpx_rms = float(np.sqrt(np.mean(mpx * mpx) + 1e-12))
            agc_rms = agc.update(mpx)  # exponential RMS estimate
            denom = max(1e-6, 3.0 * float(agc_rms))  # headroom to avoid clipping/distortion
            mpx_norm = mpx / denom
            mpx_clip_samples += int(np.sum(np.abs(mpx_norm) >= 0.999))
            mpx_total_samples += int(mpx_norm.size)
            mpx_norm = np.clip(mpx_norm, -0.999, 0.999)
            pcm_i16 = (mpx_norm * 32767.0).astype(np.int16)
            pcm_chunks.append(pcm_i16)
            total_pcm_samples += int(pcm_i16.size)
            
            now = time.time()
            
            # Periodically check if we can exit early
            if now >= last_check + check_interval_s:
                # Quick check: try to decode what we have so far
                if len(pcm_chunks) > 0 and (now - t0) >= 2.0:  # Wait at least 2 seconds
                    # Write temporary WAV and attempt decode
                    pcm_temp = np.concatenate(pcm_chunks)
                    wav_temp_path = str(Path(db_path).with_suffix(".temp_check.wav"))
                    
                    try:
                        with wave.open(wav_temp_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(fs_out)
                            wf.writeframes(pcm_temp.tobytes())
                        
                        redsea_check = RedseaDecoder(tools["redsea"])
                        groups_check, _, _, _, _ = redsea_check.decode_wav(
                            wav_temp_path,
                            rbds=True,
                            show_partial=True,
                            time_from_start=True,
                            timeout_s=5.0,
                        )
                        
                        decoded_groups_so_far = len(groups_check)
                        
                        # Clean up temp file
                        try:
                            Path(wav_temp_path).unlink()
                        except Exception:
                            pass
                        
                        # Exit early if we have enough groups
                        if decoded_groups_so_far >= min_groups_to_proceed:
                            emit_event(event_q, {
                                "type": "info",
                                "msg": f"Early exit: decoded {decoded_groups_so_far} groups at {station_freq_hz/1e6:.1f} MHz after {now-t0:.1f}s",
                                "ts": iso_utc_now(),
                            })
                            break
                            
                    except Exception:
                        pass  # Continue capturing if check fails
                
                last_check = now
            
            if now >= diag_next:
                # Pilot (19 kHz) and RDS (57 kHz) "tone-ish" SNR estimates.
                # Note: RDS is BPSK spread over a few kHz; use wider measurement bandwidth.
                try:
                    seg = mpx_norm[-min(len(mpx_norm), fs_out):].astype(np.float32, copy=False)
                    w = np.hanning(len(seg)).astype(np.float32)
                    spec = np.fft.rfft(seg * w)
                    freqs = np.fft.rfftfreq(len(seg), 1.0 / fs_out)

                    def band_power(f0: float, bw: float) -> float:
                        mask = (freqs >= (f0 - bw / 2.0)) & (freqs <= (f0 + bw / 2.0))
                        if not np.any(mask):
                            return 0.0
                        return float(np.mean(np.abs(spec[mask]) ** 2))

                    # Narrow for pilot, wider for RDS
                    pilot_p = band_power(19_000.0, 600.0)
                    rds_p = band_power(57_000.0, 4_000.0)

                    # Noise estimate excludes "interesting" bands (stereo/RDS/pilot neighborhood)
                    noise_mask = (
                        (freqs >= 2_000.0) &
                        ~((freqs >= 18_000.0) & (freqs <= 20_000.0)) &
                        ~((freqs >= 37_000.0) & (freqs <= 39_000.0)) &
                        ~((freqs >= 55_000.0) & (freqs <= 59_000.0))
                    )
                    noise_p = float(np.mean(np.abs(spec[noise_mask]) ** 2)) if np.any(noise_mask) else 1e-9

                    pilot_snr_db = 10.0 * math.log10(max(1e-12, pilot_p / max(1e-12, noise_p)))
                    rds_snr_db = 10.0 * math.log10(max(1e-12, rds_p / max(1e-12, noise_p)))
                except Exception:
                    pilot_snr_db = None
                    rds_snr_db = None

                emit_event(event_q, {
                    "type": "diag",
                    "station_freq_hz": station_freq_hz,
                    "center_freq_hz": center_freq_hz,
                    "seconds_into": now - t0,
                    "iq_rms": iq_rms,
                    "clip_pct": clip_pct / 100.0,  # keep old field semantics (0..1)
                    "iq_dc_i": iq_dc_i,
                    "iq_dc_q": iq_dc_q,
                    "pilot_snr_db": pilot_snr_db,
                    "rds_snr_db": rds_snr_db,
                    "agc_rms": float(agc_rms),
                    "mpx_rms": mpx_rms,
                    "mpx_clip_pct": (100.0 * mpx_clip_samples / max(1, mpx_total_samples)),
                    "ts": iso_utc_now(),
                })

                # Cache latest diag for final warning context
                clip_pct_latest = clip_pct / 100.0
                iq_rms_latest = iq_rms
                iq_dc_i_latest = iq_dc_i
                iq_dc_q_latest = iq_dc_q
                pilot_snr_db_latest = pilot_snr_db
                rds_snr_db_latest = rds_snr_db
                agc_rms_latest = float(agc_rms)

                diag_next = now + diag_interval_s

    except Exception as e:
        import traceback
        emit_event(event_q, {
            "type": "error",
            "msg": f"Capture loop exception: {e}",
            "traceback": traceback.format_exc(limit=12),
            "ts": iso_utc_now(),
        })
    finally:
        transfer.stop()

    elapsed = time.time() - t0

    # Concatenate PCM and write a WAV (mono, 16-bit) near the DB so users can inspect externally.
    pcm_all = np.concatenate(pcm_chunks) if pcm_chunks else np.zeros((0,), dtype=np.int16)
    wav_debug_path = str(Path(db_path).with_suffix(".last_mpx.wav"))

    try:
        with wave.open(wav_debug_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs_out)
            wf.writeframes(pcm_all.tobytes())
        emit_event(event_q, {
            "type": "debug",
            "stage": "mpx_wav_written",
            "path": wav_debug_path,
            "fs_out": fs_out,
            "n_samples": int(pcm_all.size),
            "duration_s": float(pcm_all.size) / float(fs_out) if fs_out else None,
            "ts": iso_utc_now(),
        })
    except Exception as e:
        emit_event(event_q, {"type": "error", "msg": f"Failed to write MPX WAV ({wav_debug_path}): {e}", "ts": iso_utc_now()})
        pcm_all = np.zeros((0,), dtype=np.int16)

    # Decode with redsea (offline, robust)
    decoded: list[dict] = []
    redsea = RedseaDecoder(tools["redsea"])
    redsea_timeout = max(20.0, float(cfg.seconds_per_station) * 6.0)

    if pcm_all.size > 0:
        groups, stdout_lines, stderr_lines, rc, parse_errs = redsea.decode_wav(
            wav_debug_path,
            rbds=True,
            show_partial=True,
            time_from_start=True,
            timeout_s=redsea_timeout,
        )

        # Verbose diagnostics about the decoder run (helps differentiate "no RDS present" vs pipeline problems)
        try:
            cmd_str = " ".join(getattr(redsea, "last_cmd", []) or [])
            emit_event(event_q, {
                "type": "debug",
                "stage": "redsea_done",
                "returncode": int(rc) if rc is not None else None,
                "n_groups": int(len(groups)),
                "stdout_lines": int(len(stdout_lines)),
                "stderr_lines": int(len(stderr_lines)),
                "parse_errors": int(len(parse_errs) if parse_errs is not None else 0),
                "cmd": cmd_str[:400],
                "stdout_tail": "\n".join(stdout_lines[-5:])[:800] if stdout_lines else "",
                "stderr_tail": "\n".join(stderr_lines[-5:])[:800] if stderr_lines else "",
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as _e:
            pass

        decoded = groups

        emit_event(event_q, {
            "type": "debug",
            "stage": "redsea_done",
            "cmd": " ".join(redsea.last_cmd),
            "returncode": rc,
            "stdout_lines": len(stdout_lines),
            "stderr_lines": len(stderr_lines),
            "parse_errors": parse_errs,
            "decoded_groups": len(decoded),
            "stderr_tail": "\n".join(tail_lines("\n".join(stderr_lines), 20)),
            "ts": iso_utc_now(),
        })

    # Summarize decoded groups for station_done
    latest_pi = None
    latest_ps = None
    latest_pty = None
    latest_rt = None
    for g in decoded:
        if latest_pi is None and g.get("pi"):
            latest_pi = g.get("pi")
        if latest_ps is None and g.get("ps"):
            latest_ps = g.get("ps")
        if latest_pty is None and g.get("pty") is not None:
            latest_pty = g.get("pty")
        if latest_rt is None and g.get("rt"):
            latest_rt = g.get("rt")
        if latest_pi and latest_ps and latest_rt is not None:
            break

    # Persist station summary + raw groups
    try:
        with sqlite3.connect(db_path) as conn:
            db_init(conn)
            db_upsert_station(conn, station_freq_hz, power_db, latest_pi, latest_ps, latest_pty, latest_rt)
            for g in decoded:
                db_insert_group(conn, station_freq_hz, g)
    except Exception as e:
        emit_event(event_q, {"type": "error", "msg": f"DB write failed: {e}", "ts": iso_utc_now()})

    emit_event(event_q, {
        "type": "station_done",
        "freq_hz": station_freq_hz,
        "center_freq_hz": center_freq_hz,
        "offset_hz": offset_hz,
        "power_db": power_db,
        "pi": latest_pi,
        "ps": latest_ps,
        "pty": latest_pty,
        "rt": latest_rt,
        "n_groups": len(decoded),
        "elapsed_s": elapsed,
        "target_s": float(cfg.seconds_per_station),
        "ts": iso_utc_now(),
    })

    if len(decoded) == 0:
        hackrf_tail = "\n".join(transfer.stderr_tail())
        diag_bits = f"pilot_snr_db={pilot_snr_db_latest}, rds_snr_db={rds_snr_db_latest}, clip_pct={clip_pct_latest}, iq_rms={iq_rms_latest}, agc_rms={agc_rms_latest}, iq_dc=({iq_dc_i_latest},{iq_dc_q_latest})"
        emit_event(event_q, {
            "type": "warning",
            "msg": (
                f"No RDS groups decoded at {station_freq_hz/1e6:.1f} MHz "
                f"(elapsed {elapsed:.1f}s/{float(cfg.seconds_per_station):.1f}s). Try higher gain / better antenna.\n\n"
                f"Diagnostics: {diag_bits}\n\n"
                f"MPX WAV saved: {wav_debug_path}\n\n"
                f"hackrf_transfer stderr tail:\n{hackrf_tail}\n\n"
                f"redsea stderr tail:\n{redsea.last_stderr[-1500:] if redsea.last_stderr else ''}\n"
            ),
            "ts": iso_utc_now(),
        })
    
    # Return success status
    return (len(decoded) > 0, len(decoded))


def run_scan(scan_method: str,
             fm_min_mhz: float,
             fm_max_mhz: float,
             step_khz: int,
             threshold_db: float,
             max_stations: int,
             bin_w_hz: int,
             sweep_lna: int,
             sweep_vga: int,
             sweep_amp: bool,
             step_fs_hz: int,
             step_dwell_ms: int,
             step_lna: int,
             step_vga: int,
             step_amp: bool) -> pd.DataFrame:
    if scan_method == "Fast sweep (hackrf_sweep)":
        df = run_hackrf_sweep(fm_min_mhz, fm_max_mhz, bin_w_hz, sweep_lna, sweep_vga, sweep_amp)
        return detect_fm_station_peaks(df, step_khz, threshold_db, max_stations)
    else:
        return step_scan_power(
            fmin_mhz=fm_min_mhz, fmax_mhz=fm_max_mhz, step_khz=step_khz,
            sample_rate_hz=step_fs_hz, dwell_ms=step_dwell_ms,
            vga_db=step_vga, lna_db=step_lna, amp=step_amp,
            max_stations=max_stations, threshold_db_above_median=threshold_db
        )


# ==========================
# Streamlit UI
# ==========================



class MonitorController:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.event_q: "queue.Queue[dict]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running = False

    def start(self, station_list: List[Tuple[int, float]], capture_fs: int, decim: int,
              vga: int, lna: int, amp: bool, seconds_per_station: float, offset_hz: int,
              continuous: bool = True):
        if self.running:
            return
        self._stop_evt.clear()
        self.running = True
        tool_overrides = get_tool_overrides()

        def run():
            try:
                cycle_count = 0
                while True:  # Infinite loop for continuous scanning
                    cycle_count += 1
                    self.event_q.put({
                        "type": "status",
                        "msg": f"Starting scan cycle #{cycle_count}...",
                        "ts": now_utc_iso()
                    })
                    
                    for freq_hz, power_db in station_list:
                        if self._stop_evt.is_set():
                            break
                        
                        self.event_q.put({
                            "type": "status",
                            "msg": f"Cycle #{cycle_count}: Tuning {fmt_freq(freq_hz)}…",
                            "ts": now_utc_iso()
                        })
                        
                        fs_out = int(capture_fs) // max(1, int(decim))
                        cutoff_hz = int(min(160_000, 0.49 * fs_out))
                        # Quantize/clamp gains to HackRF supported ranges
                        lna_req = int(lna)
                        vga_req = int(vga)
                        lna_applied = clamp_hackrf_lna_db(lna_req)
                        vga_applied = clamp_hackrf_vga_db(vga_req)
                        if (lna_applied != lna_req) or (vga_applied != vga_req):
                            self.event_q.put({
                                "type": "debug",
                                "stage": "gain_quantize",
                                "requested": {"lna_db": lna_req, "vga_db": vga_req},
                                "applied": {"lna_db": lna_applied, "vga_db": vga_applied},
                                "ts": now_utc_iso(),
                            })
                        cfg = CaptureConfig(
                            station_freq_hz=int(freq_hz),
                            center_freq_hz=int(freq_hz) + int(offset_hz),
                            offset_hz=int(offset_hz),
                            sample_rate_hz=int(capture_fs),
                            decim=int(decim),
                            cutoff_hz=cutoff_hz,
                            lna_db=int(lna_applied),
                            vga_db=int(vga_applied),
                            amp=bool(amp),
                            seconds=float(seconds_per_station),
                            tool_overrides=tool_overrides,
                        )

                        success, n_groups = decode_rds_for_station(
                            cfg, self.event_q, self.db_path, power_db=float(power_db)
                        )
                        
                        if success:
                            self.event_q.put({
                                "type": "info",
                                "msg": f"✓ {fmt_freq(freq_hz)}: decoded {n_groups} groups",
                                "ts": now_utc_iso()
                            })
                    
                    if self._stop_evt.is_set():
                        break
                        
                    if not continuous:
                        break
                        
                    # Brief pause between cycles
                    time.sleep(2.0)
                    
            finally:
                self.event_q.put({
                    "type": "status",
                    "msg": "Monitor loop stopped.",
                    "ts": now_utc_iso()
                })
                self.running = False

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self.running = False

class NightlyScheduler:
    """
    In-app nightly job scheduler (works only while Streamlit server stays running).
    For reliability, use Windows Task Scheduler + fm_nightly_scan.py instead.
    """
    def __init__(self, db_path: str, event_q: "queue.Queue[dict]"):
        self.db_path = db_path
        self.event_q = event_q
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self.enabled = False

    def start(self, hour: int, minute: int,
              scan_method: str,
              scan_params: dict,
              decode_top_n: int,
              decode_params: dict):
        if self._thread and self._thread.is_alive():
            return
        self.enabled = True
        self._stop_evt.clear()

        def run():
            while not self._stop_evt.is_set():
                now = now_local()
                nxt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if nxt <= now:
                    nxt = nxt + timedelta(days=1)
                sleep_s = max(1.0, (nxt - now).total_seconds())
                self.event_q.put({"type": "status", "msg": f"Nightly scan scheduled for {nxt.isoformat(timespec='minutes')}", "ts": now_utc_iso()})

                while sleep_s > 0 and not self._stop_evt.is_set():
                    chunk = min(30.0, sleep_s)
                    time.sleep(chunk)
                    sleep_s -= chunk
                if self._stop_evt.is_set():
                    break

                try:
                    self.event_q.put({"type": "status", "msg": "Nightly scan starting…", "ts": now_utc_iso()})
                    cand = run_scan(scan_method, **scan_params)
                    self.event_q.put({"type": "status", "msg": f"Nightly scan found {len(cand)} candidates.", "ts": now_utc_iso()})

                    if decode_top_n > 0 and not cand.empty:
                        top = cand.sort_values("power_db", ascending=False).head(decode_top_n)
                        station_list = [(int(r["freq_hz"]), float(r["power_db"])) for _, r in top.iterrows()]
                        mon = MonitorController(self.db_path)
                        mon.event_q = self.event_q
                        mon.start(station_list=station_list, **decode_params)
                        while mon.running and not self._stop_evt.is_set():
                            time.sleep(1.0)

                    self.event_q.put({"type": "status", "msg": "Nightly scan finished.", "ts": now_utc_iso()})
                except Exception as e:
                    self.event_q.put({"type": "error", "msg": f"Nightly job failed: {e}", "ts": now_utc_iso()})

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self.enabled = False
        self._stop_evt.set()

def init_state():
    if "db_path" not in st.session_state:
        st.session_state.db_path = DB_PATH_DEFAULT
    if "conn" not in st.session_state:
        st.session_state.conn = db_connect(st.session_state.db_path)
        db_init(st.session_state.conn)
    if "station_candidates" not in st.session_state:
        st.session_state.station_candidates = pd.DataFrame()
    if "monitor" not in st.session_state:
        st.session_state.monitor = MonitorController(st.session_state.db_path)
    if "scheduler" not in st.session_state:
        st.session_state.scheduler = NightlyScheduler(st.session_state.db_path, st.session_state.monitor.event_q)
    if "log" not in st.session_state:
        st.session_state.log = []

def drain_events() -> int:
    mon: MonitorController = st.session_state.monitor
    n = 0
    while True:
        try:
            ev = mon.event_q.get_nowait()
        except queue.Empty:
            break
        st.session_state.log.append(ev)
        st.session_state.log = st.session_state.log[-400:]
        n += 1
    return n

def tools_panel():
    st.subheader("Tools & device")

    # init overrides in session (persist across reruns)
    if "tool_overrides" not in st.session_state or not isinstance(st.session_state.tool_overrides, dict):
        st.session_state.tool_overrides = dict(TOOL_OVERRIDES_DEFAULT)

    with st.expander("Tool paths (optional overrides)", expanded=False):
        st.caption("Leave blank to use PATH. If you built redsea from source, point to redsea.exe (or drop it in C:\\HackRF\\bin and add that folder to PATH).")
        for key in DEFAULT_TOOL_KEYS:
            st.session_state.tool_overrides[key] = st.text_input(
                key,
                value=st.session_state.tool_overrides.get(key, ""),
                placeholder="(use PATH)",
            )

        c1, c2 = st.columns(2)
        if c1.button("💾 Save tool paths"):
            try:
                save_tool_overrides(st.session_state.tool_overrides)
                st.success(f"Saved: {TOOLS_CONFIG_FILE}")
            except Exception as e:
                st.error(f"Failed to save tool config: {e}")

        if c2.button("🧹 Clear overrides"):
            st.session_state.tool_overrides = {}
            try:
                save_tool_overrides({})
            except Exception:
                pass
            st.success("Cleared overrides. Using PATH again.")

    tools = {
        "hackrf_transfer": resolve_tool("hackrf_transfer", ["hackrf_transfer", "hackrf_transfer.exe"]),
        "hackrf_sweep": resolve_tool("hackrf_sweep", ["hackrf_sweep", "hackrf_sweep.exe"]),
        "hackrf_info": resolve_tool("hackrf_info", ["hackrf_info", "hackrf_info.exe"]),
        "redsea": resolve_tool("redsea", ["redsea", "redsea.exe"]),
    }
    st.write({k: (v if v else "❌ not found") for k, v in tools.items()})

    if not tools["redsea"]:
        st.warning("redsea is missing — RDS decode won’t work until redsea.exe is installed (or set above).")

    if st.button("🔌 Test HackRF connection"):
        exe = tools["hackrf_info"]
        if not exe:
            st.error("hackrf_info.exe not found. (Usually shipped with HackRF tools.)")
        else:
            try:
                cp = subprocess.run([exe], capture_output=True, text=True, timeout=10)
                out = (cp.stdout or "") + (cp.stderr or "")
                if cp.returncode == 0:
                    st.success("HackRF detected.")
                    st.code(out.strip()[:4000])
                else:
                    st.error("hackrf_info returned an error (driver/busy device?).")
                    st.code(out.strip()[:4000])
            except Exception as e:
                st.error(f"hackrf_info failed: {e}")

def scheduler_panel(conn: sqlite3.Connection):
    st.subheader("Background schedule")
    enabled = db_get_setting(conn, "nightly_enabled", "0") == "1"
    hour = int(db_get_setting(conn, "nightly_hour", str(NIGHTLY_HOUR_DEFAULT)))
    minute = int(db_get_setting(conn, "nightly_minute", str(NIGHTLY_MINUTE_DEFAULT)))
    decode_top = int(db_get_setting(conn, "nightly_decode_top", "8"))

    enabled_ui = st.checkbox("Enable in-app nightly scan (experimental)", value=enabled)
    c1, c2, c3 = st.columns(3)
    with c1:
        hour_ui = st.number_input("Nightly hour (local)", 0, 23, value=hour, step=1)
    with c2:
        minute_ui = st.number_input("Nightly minute", 0, 59, value=minute, step=1)
    with c3:
        decode_top_ui = st.number_input("Decode top N after scan", 0, 30, value=decode_top, step=1)

    if st.button("💾 Save schedule settings"):
        db_set_setting(conn, "nightly_enabled", "1" if enabled_ui else "0")
        db_set_setting(conn, "nightly_hour", str(int(hour_ui)))
        db_set_setting(conn, "nightly_minute", str(int(minute_ui)))
        db_set_setting(conn, "nightly_decode_top", str(int(decode_top_ui)))
        st.success("Saved.")

    st.info("For reliability, you can also use Windows Task Scheduler. A ready-to-run CLI script is included: fm_nightly_scan.py")

def scan_panel():
    st.subheader("Scan FM band")
    conn: sqlite3.Connection = st.session_state.conn

    sweep_exe = resolve_tool("hackrf_sweep", ["hackrf_sweep", "hackrf_sweep.exe"])
    sweep_ok, sweep_detail = hackrf_sweep_dependency_status(sweep_exe or "")
    scan_options = ["Step scan (no hackrf_sweep)"]
    if sweep_exe and sweep_ok:
        scan_options = ["Fast sweep (hackrf_sweep)", "Step scan (no hackrf_sweep)"]
    scan_method = st.selectbox("Scan method", scan_options, index=0)
    if sweep_exe and not sweep_ok:
        st.warning(
            "Fast sweep is disabled because hackrf_sweep depends on FFTW on Windows and the DLL is missing: "
            "libfftw3f-3.dll. Copy that DLL next to hackrf_sweep.exe, or use Step scan."
        )


    c1, c2, c3 = st.columns(3)
    with c1:
        fm_min = st.number_input("FM min (MHz)", value=float(FM_MIN_MHZ_DEFAULT), step=0.1)
        fm_max = st.number_input("FM max (MHz)", value=float(FM_MAX_MHZ_DEFAULT), step=0.1)
    with c2:
        step_khz = st.number_input("FM channel step (kHz)", value=int(FM_STEP_KHZ_DEFAULT), step=10)
        max_st = st.slider("Max stations to keep", 5, 60, int(MAX_STATIONS_DEFAULT), 1)
    with c3:
        th_db = st.slider("Peak threshold above median noise (dB)", 3.0, 25.0, float(PEAK_THRESHOLD_DB_DEFAULT), 0.5)

    with st.expander("Scan tuning (gains, bin width, dwell)"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            bin_w = st.number_input("Sweep bin width (Hz)", value=int(SWEEP_BIN_WIDTH_HZ_DEFAULT), step=10_000)
        with sc2:
            sweep_lna = st.number_input("Sweep LNA (dB)", value=int(SWEEP_LNA_DB_DEFAULT), step=8)
            sweep_vga = st.number_input("Sweep VGA (dB)", value=int(SWEEP_VGA_DB_DEFAULT), step=2)
        with sc3:
            sweep_amp = st.checkbox("Sweep RF amp", value=bool(SWEEP_AMP_DEFAULT))

        st.divider()
        st.caption("Step scan settings (fallback)")
        st_fs = st.number_input("Step scan sample rate (Hz)", value=2_000_000, step=100_000)
        dwell_ms = st.number_input("Step scan dwell (ms)", value=int(STEP_SCAN_DWELL_MS_DEFAULT), step=10, min_value=20, max_value=400)
        step_lna = st.number_input("Step scan LNA (dB)", value=int(TRANSFER_LNA_DB_DEFAULT), step=8)
        step_vga = st.number_input("Step scan VGA (dB)", value=int(TRANSFER_VGA_DB_DEFAULT), step=2)
        step_amp = st.checkbox("Step scan RF amp", value=bool(TRANSFER_AMP_DEFAULT))

    if st.button("🔎 Run scan", type="primary"):
        params = dict(
            fm_min_mhz=float(fm_min),
            fm_max_mhz=float(fm_max),
            step_khz=int(step_khz),
            threshold_db=float(th_db),
            max_stations=int(max_st),
            bin_w_hz=int(bin_w),
            sweep_lna=int(sweep_lna),
            sweep_vga=int(sweep_vga),
            sweep_amp=bool(sweep_amp),
            step_fs_hz=int(st_fs),
            step_dwell_ms=int(dwell_ms),
            step_lna=int(step_lna),
            step_vga=int(step_vga),
            step_amp=bool(step_amp),
        )
        with st.spinner("Scanning…"):
            try:
                cand = run_scan(scan_method, **params)
                st.session_state.station_candidates = cand
                st.success(f"Found {len(cand)} station candidates.")
            except Exception as e:
                st.error(str(e))

    if not st.session_state.station_candidates.empty:
        st.markdown("### Station candidates")
        st.dataframe(st.session_state.station_candidates, width='stretch', hide_index=True)

    if st.button("📌 Save current scan settings for nightly job"):
        db_set_setting(conn, "scan_method", scan_method)
        # Save last used params if present
        try:
            db_set_setting(conn, "scan_params", json.dumps(params))
        except Exception:
            pass
        st.success("Saved scan config for nightly job.")

def monitor_panel():
    st.subheader("Decode RDS")
    cand = st.session_state.station_candidates
    if cand.empty:
        st.info("Run a scan first, or pick stations from your Guide.")
        return

    selected_freqs = st.multiselect(
        "Frequencies to decode (MHz)",
        options=[float(x) for x in cand["freq_mhz"].tolist()],
        default=[float(x) for x in cand["freq_mhz"].tolist()[: min(10, len(cand))]],
    )
    selected = [(int(round(mhz * 1e6)), float(cand.loc[cand["freq_mhz"] == mhz, "power_db"].values[0]))
                for mhz in selected_freqs]

    c1, c2, c3 = st.columns(3)
    with c1:
        cap_fs = st.number_input("Capture sample rate (Hz)", value=int(CAPTURE_FS_HZ_DEFAULT), step=10_000)
        decim = st.number_input("Decimation factor", value=int(CAPTURE_DECIM_DEFAULT), step=1, min_value=2, max_value=50)
    with c2:
        vga = st.number_input("Transfer VGA (dB)", value=int(TRANSFER_VGA_DB_DEFAULT), step=2)
        lna = st.number_input("Transfer LNA (dB)", value=int(TRANSFER_LNA_DB_DEFAULT), step=8)
        amp = st.checkbox("Transfer RF amp", value=bool(TRANSFER_AMP_DEFAULT))
    with c3:
        sec = st.number_input("Max seconds per station", value=float(CAPTURE_SECONDS_DEFAULT), step=1.0, min_value=1.0, max_value=30.0)
        offset_hz = st.number_input(
            "Offset tune (Hz)",
            min_value=0,
            max_value=500000,
            value=250000,
            step=50000,
            help="Tunes to station+offset and digitally mixes down. Helps reject DC spike and improves RDS.",
        )

    continuous_mode = st.checkbox(
        "🔄 Continuous scanning mode",
        value=True,
        help="Keep cycling through all stations. Moves to next station once RDS is decoded."
    )

    fs_out = int(cap_fs) // int(decim)
    st.caption(f"MPX sample rate to redsea: ~{fs_out} Hz (should be >114 kHz for RDS @57 kHz).")

    mon: MonitorController = st.session_state.monitor
    colA, colB = st.columns(2)
    with colA:
        if st.button("📡 Start decoding", type="primary", disabled=mon.running):
            mon.start(
                selected, int(cap_fs), int(decim), int(vga), int(lna),
                bool(amp), float(sec), int(offset_hz), continuous=continuous_mode
            )
            st.toast("Decoding started in continuous mode." if continuous_mode else "Decoding started.")
    with colB:
        if st.button("🛑 Stop", disabled=not mon.running):
            mon.stop()
            st.toast("Stopping…")

    if st.button("📌 Save current decode settings for nightly job"):
        conn: sqlite3.Connection = st.session_state.conn
        db_set_setting(conn, "decode_params", json.dumps({
            "capture_fs": int(cap_fs),
            "decim": int(decim),
            "vga": int(vga),
            "lna": int(lna),
            "amp": bool(amp),
            "seconds_per_station": float(sec),
            "offset_hz": int(offset_hz),
        }))
        st.success("Saved decode config for nightly job.")

    drained = drain_events()
    if drained:
        st.toast(f"Updated ({drained} events)")

    last_status = next((x for x in reversed(st.session_state.log) if x.get("type") == "status"), None)
    if last_status:
        st.info(last_status.get("msg"))

    last_done = next((x for x in reversed(st.session_state.log) if x.get("type") == "station_done"), None)
    if last_done:
        el = float(last_done.get("elapsed_s") or 0.0)
        tgt = float(last_done.get("target_s") or 0.0)
        st.success(
            f"Last decoded: {fmt_freq(last_done['freq_hz'])} • groups: {last_done['n_groups']} • "
            f"elapsed: {el:.1f}s/{tgt:.1f}s • PS: {last_done.get('ps') or '-'}"
        )

    for ev in list(reversed(st.session_state.log))[:10]:
        if ev.get("type") == "warning":
            st.warning(ev.get("msg"))
            break
        if ev.get("type") == "error":
            st.error(ev.get("msg"))
            break


def timeline_panel(conn: sqlite3.Connection, freq_hz: int):
    hist = db_load_history(conn, freq_hz=freq_hz, limit=2000)
    if hist.empty:
        st.info("No RDS history yet for this station.")
        return

    hist = hist.copy()
    hist["ts_dt"] = pd.to_datetime(hist["ts"], errors="coerce")
    hist = hist.dropna(subset=["ts_dt"]).sort_values("ts_dt")
    hist["rt"] = hist["rt"].fillna("")
    changes = hist[hist["rt"].shift(1) != hist["rt"]].copy()
    if changes.empty:
        changes = hist.tail(200).copy()

    st.caption(f"Ticker events: {len(changes)} (RT change points).")
    idx = st.slider("Scrub", 0, max(0, len(changes)-1), value=max(0, len(changes)-1), step=1)
    row = changes.iloc[idx]
    ts = row["ts_dt"].astimezone(now_local().tzinfo)
    rt = row["rt"] or "(no radiotext)"
    ps = row.get("ps") or ""
    pi = row.get("pi") or ""

    st.markdown(f"### {fmt_freq(freq_hz)} — {ps}")
    st.caption(f"At {ts.isoformat(timespec='seconds')} • PI={pi}")
    st.markdown(f"#### 📜 {rt}")

    with st.expander("Raw history table"):
        view = changes[["ts","pi","ps","pty","rt"]].tail(400)
        st.dataframe(view, width='stretch', hide_index=True)

def guide_panel():
    st.subheader("Guide (your FM TVGuide)")
    conn: sqlite3.Connection = st.session_state.conn
    df = db_load_stations(conn)

    tabs = st.tabs(["Lineup", "Stations", "Timeline"])
    with tabs[0]:
        if df.empty:
            st.info("No stations yet. Run Scan + Decode to populate.")
        else:
            lineup = lineup_by_pi(df)
            q = st.text_input("Search lineup (PS / RT / PI)", value="", key="lineup_search")
            ldf = lineup.copy()
            if q.strip():
                ql = q.strip().lower()
                def match(row) -> bool:
                    for k in ("ps","rt_sample","pi","freqs_mhz"):
                        v = row.get(k)
                        if isinstance(v, str) and ql in v.lower():
                            return True
                    return False
                ldf = ldf[ldf.apply(match, axis=1)]
            st.dataframe(ldf, width='stretch', hide_index=True)
            st.caption("Lineup learning: frequencies are grouped by PI (so translators/repeaters merge automatically).")

    with tabs[1]:
        if df.empty:
            st.info("No stations yet.")
        else:
            df2 = df.copy()
            df2["freq_mhz"] = df2["freq_hz"] / 1e6
            q = st.text_input("Search stations (PS / RT / PI)", value="", key="stations_search")
            if q.strip():
                ql = q.strip().lower()
                def match(row) -> bool:
                    for k in ("ps","rt","pi"):
                        v = row.get(k)
                        if isinstance(v, str) and ql in v.lower():
                            return True
                    return False
                df2 = df2[df2.apply(match, axis=1)]
            view = df2[["freq_mhz","power_db","ps","rt","pi","pty","first_seen","last_seen"]].sort_values("freq_mhz")
            st.dataframe(view, width='stretch', hide_index=True)

        st.markdown("### Add station manually")
        c1, c2, c3 = st.columns(3)
        with c1:
            mhz = st.number_input("Frequency (MHz)", value=99.5, step=0.1)
        with c2:
            pwr = st.number_input("Power (dB)", value=-60.0, step=1.0)
        with c3:
            if st.button("➕ Add/Update station"):
                db_upsert_station(conn, int(round(float(mhz) * 1e6)), float(pwr), None, None, None, None)
                st.success("Added. Now run Decode.")

    with tabs[2]:
        if df.empty:
            st.info("No stations yet.")
        else:
            df2 = df.copy()
            df2["freq_mhz"] = df2["freq_hz"] / 1e6
            options = [float(x) for x in df2["freq_mhz"].tolist()]
            sel = st.selectbox("Pick a station (MHz)", options=options, index=0)
            freq_hz = int(round(float(sel) * 1e6))
            timeline_panel(conn, freq_hz)

def maybe_start_scheduler():
    conn: sqlite3.Connection = st.session_state.conn
    enabled = db_get_setting(conn, "nightly_enabled", "0") == "1"
    if not enabled:
        return
    if st.session_state.get("_scheduler_started", False):
        return

    hour = int(db_get_setting(conn, "nightly_hour", str(NIGHTLY_HOUR_DEFAULT)))
    minute = int(db_get_setting(conn, "nightly_minute", str(NIGHTLY_MINUTE_DEFAULT)))
    decode_top = int(db_get_setting(conn, "nightly_decode_top", "8"))

    scan_method = db_get_setting(conn, "scan_method", "Fast sweep (hackrf_sweep)")
    scan_params_json = db_get_setting(conn, "scan_params", "")
    decode_params_json = db_get_setting(conn, "decode_params", "")

    if not scan_params_json:
        scan_params = dict(
            fm_min_mhz=FM_MIN_MHZ_DEFAULT, fm_max_mhz=FM_MAX_MHZ_DEFAULT,
            step_khz=FM_STEP_KHZ_DEFAULT, threshold_db=PEAK_THRESHOLD_DB_DEFAULT, max_stations=MAX_STATIONS_DEFAULT,
            bin_w_hz=SWEEP_BIN_WIDTH_HZ_DEFAULT, sweep_lna=SWEEP_LNA_DB_DEFAULT, sweep_vga=SWEEP_VGA_DB_DEFAULT, sweep_amp=SWEEP_AMP_DEFAULT,
            step_fs_hz=2_000_000, step_dwell_ms=STEP_SCAN_DWELL_MS_DEFAULT, step_lna=TRANSFER_LNA_DB_DEFAULT, step_vga=TRANSFER_VGA_DB_DEFAULT, step_amp=TRANSFER_AMP_DEFAULT
        )
    else:
        scan_params = json.loads(scan_params_json)

    if not decode_params_json:
        decode_params = dict(
            capture_fs=CAPTURE_FS_HZ_DEFAULT, decim=CAPTURE_DECIM_DEFAULT,
            vga=TRANSFER_VGA_DB_DEFAULT, lna=TRANSFER_LNA_DB_DEFAULT, amp=TRANSFER_AMP_DEFAULT,
            seconds_per_station=CAPTURE_SECONDS_DEFAULT
        )
    else:
        decode_params = json.loads(decode_params_json)
        decode_params["offset_hz"] = int(decode_params.get("offset_hz", 250000))

    st.session_state.scheduler.start(
        hour=hour, minute=minute,
        scan_method=scan_method,
        scan_params=scan_params,
        decode_top_n=decode_top,
        decode_params=decode_params
    )
    st.session_state["_scheduler_started"] = True


def main():
    st.set_page_config(page_title="FM RDS Guide", layout="wide")
    init_state()
    maybe_start_scheduler()

    st.title("📻 FM RDS Guide")
    st.caption("A TVGuide-style table of contents for your local FM band — scan, decode RDS, and browse history like a ticker.")

    with st.sidebar:
        tools_panel()
        st.divider()
        st.markdown("### Database")
        db_path = st.text_input("SQLite DB path", value=st.session_state.db_path)
        if db_path != st.session_state.db_path:
            st.session_state.db_path = db_path
            try:
                st.session_state.conn.close()
            except Exception:
                pass
            st.session_state.conn = db_connect(db_path)
            db_init(st.session_state.conn)
            st.session_state.monitor = MonitorController(db_path)
            st.session_state.scheduler = NightlyScheduler(db_path, st.session_state.monitor.event_q)
            st.session_state.log = []
            st.session_state["_scheduler_started"] = False
            st.success("Switched DB.")

        st.divider()
        scheduler_panel(st.session_state.conn)

    tabs = st.tabs(["Scan", "Decode", "Guide", "Debug log"])
    with tabs[0]:
        scan_panel()
    with tabs[1]:
        monitor_panel()
    with tabs[2]:
        guide_panel()
    with tabs[3]:
        drain_events()
        st.json(st.session_state.log[-120:])

if __name__ == "__main__":
    main()
