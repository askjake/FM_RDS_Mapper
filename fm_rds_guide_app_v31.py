# -*- coding: utf-8 -*-
"""
fm_rds_guide_app_v31.py

Streamlit UI for scanning FM band and decoding RDS (via HackRF + redsea).

Run:
  streamlit run fm_rds_guide_app_v31.py
"""
from __future__ import annotations

import queue
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from Defaults import FM_MAX_MHZ_DEFAULT, FM_MIN_MHZ_DEFAULT, ToolPaths, make_capture_cfg
from FM_demod_RDS_decode import auto_tune_station, decode_rds_for_station
from Scan_methods import run_hackrf_sweep, sweep_to_channel_peaks
from SQLite_storage import get_best_cfg, get_recent_runs, list_stations, insert_run, upsert_station
from Utilities import emit_event, resolve_tool

APP_TITLE = "FM RDS Guide (v31)"

def _init_state():
    if "event_q" not in st.session_state:
        st.session_state.event_q = queue.Queue()
    if "events" not in st.session_state:
        st.session_state.events = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "toolpaths" not in st.session_state:
        st.session_state.toolpaths = ToolPaths()
    if "db_path" not in st.session_state:
        st.session_state.db_path = str(Path.cwd() / "fm_rds_guide.sqlite3")
    if "scan_df" not in st.session_state:
        st.session_state.scan_df = None

def _drain_events(max_items: int = 200):
    q = st.session_state.event_q
    events = st.session_state.events
    for _ in range(max_items):
        try:
            events.append(q.get_nowait())
        except Exception:
            break
    # keep tail
    st.session_state.events = events[-600:]

def _tool_status(tp: ToolPaths) -> Dict[str, Optional[str]]:
    return {
        "hackrf_transfer": resolve_tool(tp.hackrf_transfer, ["hackrf_transfer", "hackrf_transfer.exe"]),
        "hackrf_sweep": resolve_tool(tp.hackrf_sweep, ["hackrf_sweep", "hackrf_sweep.exe"]),
        "redsea": resolve_tool(tp.redsea, ["redsea", "redsea.exe"]),
    }

def _render_event_log():
    _drain_events()
    ev = st.session_state.events
    if not ev:
        st.info("No events yet.")
        return
    # show as text for simplicity
    lines = []
    for e in ev[-120:]:
        ts = e.get("ts", "")
        stage = e.get("stage", e.get("type", ""))
        msg = e.get("msg", "")
        if not msg and e.get("type") == "error":
            msg = e.get("err", "")
        lines.append(f"{ts} | {stage} | {msg}".rstrip())
    st.code("\n".join(lines), language="text")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _init_state()

    st.title(APP_TITLE)
    st.caption("HackRF + FM composite (MPX) capture + RDS decode via redsea. Stores history in SQLite and writes debug WAVs next to the DB.")

    # Sidebar: tools & DB
    with st.sidebar:
        st.header("Tools & storage")
        tp: ToolPaths = st.session_state.toolpaths

        tp.hackrf_transfer = st.text_input("hackrf_transfer path/name", tp.hackrf_transfer)
        tp.hackrf_sweep = st.text_input("hackrf_sweep path/name", tp.hackrf_sweep)
        tp.redsea = st.text_input("redsea path/name", tp.redsea)

        st.session_state.db_path = st.text_input("SQLite DB path", st.session_state.db_path)

        status = _tool_status(tp)
        st.subheader("Detected")
        st.write({k: (v or "NOT FOUND") for k, v in status.items()})

        st.session_state.toolpaths = tp

    # Main controls
    colA, colB = st.columns([1.2, 1.0], gap="large")

    with colA:
        st.subheader("Station")
        freq_mhz = st.number_input("Station frequency (MHz)", min_value=50.0, max_value=300.0, value=101.1, step=0.1)

        # Load best cfg if present
        best_cfg = get_best_cfg(st.session_state.db_path, int(round(freq_mhz*1e6)))
        if best_cfg:
            st.success("Found saved best config for this station in the DB. You can override values below.")
        else:
            st.info("No saved config for this station yet. Use defaults or Auto-tune.")

        st.subheader("Capture / decode settings")
        base = make_capture_cfg(freq_mhz)
        if best_cfg:
            # apply best cfg fields if compatible
            for k, v in best_cfg.items():
                if hasattr(base, k):
                    try:
                        setattr(base, k, v)
                    except Exception:
                        pass

        # Editable fields
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            offset_khz = st.number_input("Offset (kHz)", value=float(base.offset_hz)/1000.0, step=1.0)
            seconds = st.number_input("Capture seconds", value=float(base.seconds), min_value=2.0, max_value=60.0, step=1.0)
        with c2:
            sample_rate = st.selectbox("Sample rate (Hz)", options=[2_052_000, 2_280_000, 2_400_000, 5_000_000], index=1 if base.sample_rate_hz==2_280_000 else 0)
            decim = st.selectbox("Decimation", options=[10, 12, 13, 16], index=1)
        with c3:
            lna_db = st.selectbox("LNA (dB)", options=[0,8,16,24,32,40], index=5)
            vga_db = st.slider("VGA (dB)", min_value=0, max_value=62, value=int(base.vga_db), step=2)
        with c4:
            amp = st.checkbox("AMP (-a)", value=bool(base.amp))
            redsea_input = st.selectbox("Redsea input", options=["full", "rds_bandpass"], index=0)

        # Build cfg
        cfg = make_capture_cfg(
            freq_mhz,
            offset_hz=int(offset_khz * 1000),
            sample_rate_hz=int(sample_rate),
            decim=int(decim),
            lna_db=int(lna_db),
            vga_db=int(vga_db),
            amp=bool(amp),
            seconds=float(seconds),
            redsea_input=str(redsea_input),
        )

        st.write("Derived:", {
            "center_mhz": cfg.center_freq_hz/1e6,
            "fs_out_hz": cfg.fs_out,
            "cutoff_hz": cfg.cutoff_hz,
            "nco_hz": cfg.nco_freq_hz,
        })

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Decode station now", type="primary"):
                try:
                    with st.spinner("Capturing + decoding…"):
                        r = decode_rds_for_station(cfg, st.session_state.toolpaths, event_q=st.session_state.event_q, db_path=st.session_state.db_path, power_db=None)
                    st.session_state.last_result = r
                    insert_run(st.session_state.db_path, r["station_freq_hz"], r)
                    upsert_station(st.session_state.db_path, r["station_freq_hz"], r, best_cfg=r.get("cfg"))
                except Exception as e:
                    st.error(str(e))
        with b2:
            if st.button("Auto-tune this station"):
                try:
                    with st.spinner("Auto-tuning (multiple captures)…"):
                        base_cfg = cfg
                        base_cfg.seconds = float(seconds)  # user-chosen
                        r = auto_tune_station(float(freq_mhz), st.session_state.toolpaths, event_q=st.session_state.event_q, base_cfg=base_cfg, power_db=None)
                    st.session_state.last_result = r
                    insert_run(st.session_state.db_path, r["station_freq_hz"], r | {"notes": "autotune_best"})
                    upsert_station(st.session_state.db_path, r["station_freq_hz"], r, best_cfg=r.get("cfg"))
                except Exception as e:
                    st.error(str(e))

        st.subheader("Latest result")
        r = st.session_state.last_result
        if r:
            st.write({
                "n_groups": r.get("n_groups"),
                "PI": r.get("pi"),
                "PS": r.get("ps"),
                "PTY": r.get("pty"),
                "RT": r.get("rt"),
                "pilot_snr_db": r.get("pilot_snr_db"),
                "rds_snr_db": r.get("rds_snr_db"),
                "clip_pct": r.get("clip_pct"),
                "elapsed_s": r.get("elapsed_s"),
                "wav_path": r.get("wav_path"),
            })
            if r.get("redsea", {}).get("groups_preview"):
                st.write("redsea preview (first 10 JSON lines):")
                st.json(r["redsea"]["groups_preview"], expanded=False)
            if r.get("redsea", {}).get("stderr_tail"):
                st.write("redsea stderr tail:")
                st.code(r["redsea"]["stderr_tail"], language="text")
        else:
            st.info("No decode yet.")

    with colB:
        st.subheader("Scan FM band (optional)")
        c1, c2, c3 = st.columns(3)
        with c1:
            fmin = st.number_input("Scan start (MHz)", value=float(FM_MIN_MHZ_DEFAULT), step=0.1)
        with c2:
            fmax = st.number_input("Scan stop (MHz)", value=float(FM_MAX_MHZ_DEFAULT), step=0.1)
        with c3:
            scan_seconds = st.number_input("Scan seconds", value=2.5, min_value=1.0, max_value=10.0, step=0.5)

        if st.button("Run scan (hackrf_sweep)"):
            try:
                with st.spinner("Running hackrf_sweep…"):
                    bins = run_hackrf_sweep(
                        st.session_state.toolpaths.hackrf_sweep,
                        float(fmin),
                        float(fmax),
                        bin_width_hz=200_000,
                        lna_db=40,
                        vga_db=20,
                        amp=False,
                        seconds=float(scan_seconds),
                    )
                peaks = sweep_to_channel_peaks(bins)
                df = pd.DataFrame(peaks)
                st.session_state.scan_df = df
            except Exception as e:
                st.error(str(e))

        df = st.session_state.scan_df
        if isinstance(df, pd.DataFrame) and not df.empty:
            st.write("Top candidates:")
            st.dataframe(df.head(25), use_container_width=True)
            # quick plot
            try:
                st.line_chart(df.sort_values("freq_mhz").set_index("freq_mhz")["power_db"])
            except Exception:
                pass
        else:
            st.info("No scan data yet.")

        st.subheader("Database view")
        try:
            stations = list_stations(st.session_state.db_path, limit=50)
            if stations:
                st.dataframe(pd.DataFrame(stations), use_container_width=True)
            runs = get_recent_runs(st.session_state.db_path, limit=50)
            if runs:
                st.write("Recent runs:")
                st.dataframe(pd.DataFrame(runs), use_container_width=True)
        except Exception as e:
            st.warning(f"DB read issue: {e}")

        st.subheader("Event log")
        _render_event_log()

if __name__ == "__main__":
    main()