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
from SQLite_storage import *
from Scan_methods import *
from FM_demod_RDS_decode import *
from Utilities import *
from Defaults import *

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
              vga: int, lna: int, amp: bool, seconds_per_station: float, offset_hz: int):
        if self.running:
            return
        self._stop_evt.clear()
        self.running = True
        tool_overrides = get_tool_overrides()

        def run():
            try:
                for freq_hz, power_db in station_list:
                    if self._stop_evt.is_set():
                        break
                    self.event_q.put({"type": "status", "msg": f"Tuning {fmt_freq(freq_hz)}…", "ts": now_utc_iso()})
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
                    # Guard rails: HackRF needs >= 2 MHz RF sample rate; redsea needs MPX WAV >= 128 kHz.
                    if int(capture_fs) < 2_000_000:
                        raise ValueError(f"Capture sample rate must be >= 2_000_000 Hz (got {capture_fs}).")
                    fs_out_check = int(capture_fs) // int(decim)
                    if fs_out_check < 128_000:
                        raise ValueError(f"Post-decimation MPX rate too low for redsea: fs_out={fs_out_check} Hz (need >= 128_000). Reduce decim or increase capture sample rate.")

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

                    decode_rds_for_station(cfg, self.event_q, self.db_path, power_db=float(power_db))
            finally:
                self.event_q.put({"type": "status", "msg": "Monitor loop stopped.", "ts": now_utc_iso()})
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
        cap_fs = st.number_input("Capture sample rate (Hz)", min_value=2_000_000, value=int(CAPTURE_FS_HZ_DEFAULT), step=10_000)
        decim = st.number_input("Decimation factor", value=int(CAPTURE_DECIM_DEFAULT), step=1, min_value=2, max_value=50)
    with c2:
        vga = st.number_input("Transfer VGA (dB)", value=int(TRANSFER_VGA_DB_DEFAULT), step=2)
        lna = st.number_input("Transfer LNA (dB)", value=int(TRANSFER_LNA_DB_DEFAULT), step=8)
        amp = st.checkbox("Transfer RF amp", value=bool(TRANSFER_AMP_DEFAULT))
    with c3:
        sec = st.number_input("Seconds per station", value=float(CAPTURE_SECONDS_DEFAULT), step=1.0, min_value=1.0, max_value=30.0)
        offset_hz = st.number_input(
            "Offset tune (Hz)",
            min_value=0,
            max_value=500000,
            value=250000,
            step=50000,
            help="Tunes to station+offset and digitally mixes down. Helps reject DC spike and improves RDS.",
        )

    fs_out = int(cap_fs) // int(decim)
    st.caption(f"MPX sample rate to redsea: ~{fs_out} Hz (should be >114 kHz for RDS @57 kHz).")

    mon: MonitorController = st.session_state.monitor
    colA, colB = st.columns(2)
    with colA:
        if st.button("📡 Start decoding", type="primary", disabled=mon.running):
            mon.start(selected, int(cap_fs), int(decim), int(vga), int(lna), bool(amp), float(sec), int(offset_hz))
            st.toast("Decoding started.")
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