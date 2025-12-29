# -*- coding: utf-8 -*-
"""
SQLite_storage.py (v31)

SQLite persistence for station results and run history.
Schema auto-creates and is forward-compatible.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30, isolation_level=None)  # autocommit
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def _table_columns(con, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}

def ensure_schema(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        cols = _table_columns(con, "stations")
        if "best_cfg_json" not in cols:
            con.execute("ALTER TABLE stations ADD COLUMN best_cfg_json TEXT")
            con.commit()
    finally:
        con.close()

def init_db(db_path: str) -> None:
    con = _connect(db_path)
    try:
        con.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            freq_hz INTEGER PRIMARY KEY,
            last_seen_utc INTEGER,
            power_db REAL,
            pi TEXT,
            ps TEXT,
            pty INTEGER,
            rt TEXT,
            n_groups INTEGER,
            elapsed_s REAL,
            best_cfg_json TEXT
        );
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS station_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc INTEGER,
            freq_hz INTEGER,
            center_freq_hz INTEGER,
            power_db REAL,
            n_groups INTEGER,
            pilot_snr_db REAL,
            rds_snr_db REAL,
            clip_pct REAL,
            iq_rms REAL,
            agc_rms REAL,
            cfg_json TEXT,
            notes TEXT
        );
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_station_runs_freq_ts ON station_runs(freq_hz, ts_utc);")
    finally:
        con.close()

def upsert_station(db_path: str, freq_hz: int, payload: Dict[str, Any], best_cfg: Optional[Dict[str, Any]] = None) -> None:
    init_db(db_path)
    con = _connect(db_path)
    try:
        now = int(time.time())
        pi = payload.get("pi")
        ps = payload.get("ps")
        pty = payload.get("pty")
        rt = payload.get("rt")
        power_db = payload.get("power_db")
        n_groups = int(payload.get("n_groups") or 0)
        elapsed_s = payload.get("elapsed_s", None)
        best_cfg_json = json.dumps(best_cfg or payload.get("best_cfg") or payload.get("cfg") or {}, ensure_ascii=False)
        con.execute("""
        INSERT INTO stations(freq_hz, last_seen_utc, power_db, pi, ps, pty, rt, n_groups, elapsed_s, best_cfg_json)
        VALUES(?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(freq_hz) DO UPDATE SET
            last_seen_utc=excluded.last_seen_utc,
            power_db=excluded.power_db,
            pi=COALESCE(excluded.pi, stations.pi),
            ps=COALESCE(excluded.ps, stations.ps),
            pty=COALESCE(excluded.pty, stations.pty),
            rt=COALESCE(excluded.rt, stations.rt),
            n_groups=excluded.n_groups,
            elapsed_s=excluded.elapsed_s,
            best_cfg_json=excluded.best_cfg_json
        """, (int(freq_hz), now, power_db, pi, ps, pty, rt, n_groups, elapsed_s, best_cfg_json))
    finally:
        con.close()

def insert_run(db_path: str, freq_hz: int, run_payload: Dict[str, Any]) -> None:
    init_db(db_path)
    con = _connect(db_path)
    try:
        con.execute("""
        INSERT INTO station_runs(
            ts_utc, freq_hz, center_freq_hz, power_db, n_groups,
            pilot_snr_db, rds_snr_db, clip_pct, iq_rms, agc_rms,
            cfg_json, notes
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            int(time.time()),
            int(freq_hz),
            int(run_payload.get("center_freq_hz") or 0),
            run_payload.get("power_db"),
            int(run_payload.get("n_groups") or 0),
            run_payload.get("pilot_snr_db"),
            run_payload.get("rds_snr_db"),
            run_payload.get("clip_pct"),
            run_payload.get("iq_rms"),
            run_payload.get("agc_rms"),
            json.dumps(run_payload.get("cfg") or {}, ensure_ascii=False),
            run_payload.get("notes"),
        ))
    finally:
        con.close()

def get_best_cfg(db_path: str, freq_hz: int) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    con = _connect(db_path)
    try:
        cur = con.execute("SELECT best_cfg_json FROM stations WHERE freq_hz=?", (int(freq_hz),))
        row = cur.fetchone()
        if not row or not row[0]:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None
    finally:
        con.close()

def list_stations(db_path: str, limit: int = 200) -> List[Dict[str, Any]]:
    init_db(db_path)
    con = _connect(db_path)
    try:
        cur = con.execute("""
            SELECT freq_hz, last_seen_utc, power_db, pi, ps, pty, rt, n_groups, elapsed_s
            FROM stations
            ORDER BY last_seen_utc DESC
            LIMIT ?
        """, (int(limit),))
        out = []
        for r in cur.fetchall():
            out.append({
                "freq_hz": r[0],
                "freq_mhz": r[0] / 1e6,
                "last_seen_utc": r[1],
                "power_db": r[2],
                "pi": r[3],
                "ps": r[4],
                "pty": r[5],
                "rt": r[6],
                "n_groups": r[7],
                "elapsed_s": r[8],
            })
        return out
    finally:
        con.close()

def get_recent_runs(db_path: str, freq_hz: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
    init_db(db_path)
    con = _connect(db_path)
    try:
        if freq_hz is None:
            cur = con.execute("""
                SELECT ts_utc, freq_hz, center_freq_hz, power_db, n_groups, pilot_snr_db, rds_snr_db, clip_pct, iq_rms, agc_rms, notes
                FROM station_runs
                ORDER BY ts_utc DESC
                LIMIT ?
            """, (int(limit),))
        else:
            cur = con.execute("""
                SELECT ts_utc, freq_hz, center_freq_hz, power_db, n_groups, pilot_snr_db, rds_snr_db, clip_pct, iq_rms, agc_rms, notes
                FROM station_runs
                WHERE freq_hz=?
                ORDER BY ts_utc DESC
                LIMIT ?
            """, (int(freq_hz), int(limit)))
        out = []
        for r in cur.fetchall():
            out.append({
                "ts_utc": r[0],
                "freq_hz": r[1],
                "freq_mhz": r[1] / 1e6,
                "center_freq_hz": r[2],
                "center_mhz": r[2] / 1e6,
                "power_db": r[3],
                "n_groups": r[4],
                "pilot_snr_db": r[5],
                "rds_snr_db": r[6],
                "clip_pct": r[7],
                "iq_rms": r[8],
                "agc_rms": r[9],
                "notes": r[10],
            })
        return out
    finally:
        con.close()