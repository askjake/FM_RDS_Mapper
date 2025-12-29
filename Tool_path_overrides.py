# ==========================
# Tool path overrides (optional)
# ==========================
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

from Utilities import *
from Defaults import *
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

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

