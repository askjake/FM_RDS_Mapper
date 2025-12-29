# -*- coding: utf-8 -*-
"""
FM_demod_RDS_decode.py (v31)

HackRF capture + FM demod to MPX + RDS decode via redsea.

Key design choices:
- Capture to a temporary IQ file via hackrf_transfer (-r <file>), then stream-process the file.
  This is more Windows-friendly than relying on stdout pipes (some builds behave oddly).
- Digital offset tuning: tune HackRF at (station + offset), then NCO mix by -offset.
- FIR + decimation on complex IQ, then quadrature FM demod => MPX.
- Save MPX as PCM16 WAV and run redsea offline for robust parsing.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Defaults import CaptureConfig, ToolPaths, HACKRF_MIN_SAMPLE_RATE_HZ, RDS_SUBCARRIER_HZ
from Utilities import (
    FIRDecimator,
    MovingRMS,
    emit_event,
    estimate_pilot_and_rds_snr_db,
    fir_bandpass,
    fir_lowpass,
    float_to_int16_pcm,
    fm_quadrature_demod,
    parse_json_lines,
    quantize_hackrf_gain,
    resolve_tool,
    tail_lines,
)

# ==========================
# HackRF capture
# ==========================
def _run_hackrf_transfer_to_file(
    exe: str,
    *,
    center_freq_hz: int,
    sample_rate_hz: int,
    lna_db: int,
    vga_db: int,
    amp: bool,
    seconds: float,
    out_path: str,
) -> Dict[str, Any]:
    """
    Run hackrf_transfer capturing interleaved unsigned 8-bit IQ to `out_path`.

    Returns dict with cmd, returncode, stderr_tail.
    """
    if int(sample_rate_hz) < HACKRF_MIN_SAMPLE_RATE_HZ:
        raise ValueError(f"sample_rate_hz must be >= {HACKRF_MIN_SAMPLE_RATE_HZ}")

    lna_applied, vga_applied, _ = quantize_hackrf_gain(lna_db, vga_db)
    n_samples = int(float(sample_rate_hz) * float(seconds))

    cmd = [
        exe,
        "-f", str(int(center_freq_hz)),
        "-s", str(int(sample_rate_hz)),
        "-l", str(int(lna_applied)),
        "-g", str(int(vga_applied)),
        "-a", "1" if amp else "0",
        "-n", str(int(n_samples)),
        "-r", str(out_path),
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "stdout_tail": "\n".join(tail_lines((out or b"").decode("utf-8", errors="replace"), 20)),
        "stderr_tail": "\n".join(tail_lines((err or b"").decode("utf-8", errors="replace"), 40)),
        "lna_applied": lna_applied,
        "vga_applied": vga_applied,
        "n_samples": n_samples,
    }

def _u8iq_bytes_to_c64(raw: bytes) -> np.ndarray:
    """
    Convert interleaved unsigned 8-bit IQ bytes to complex64 centered around 0.
    HackRF uses unsigned 8-bit IQ (I,Q in [0..255]).
    """
    if not raw:
        return np.empty((0,), dtype=np.complex64)
    x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    i = (x[0::2] - 127.5) / 127.5
    q = (x[1::2] - 127.5) / 127.5
    n = min(i.size, q.size)
    if n <= 0:
        return np.empty((0,), dtype=np.complex64)
    return (i[:n] + 1j * q[:n]).astype(np.complex64)

def _nco_mix_chunk(x: np.ndarray, fs: float, f_hz: float, phase0: float) -> Tuple[np.ndarray, float]:
    """
    Mix by exp(+j*2pi*f*n/fs) (so if f is negative, we mix down).
    Maintains phase continuity between chunks.
    """
    if x.size == 0 or float(f_hz) == 0.0:
        return x, float(phase0)
    n = np.arange(x.size, dtype=np.float32)
    phase = float(phase0) + 2.0 * np.pi * (float(f_hz) / float(fs)) * n
    osc = np.exp(1j * phase).astype(np.complex64)
    y = x * osc
    phase_end = float(phase[-1] + 2.0 * np.pi * (float(f_hz) / float(fs)))
    phase_end = float(np.fmod(phase_end, 2.0 * np.pi))
    return y, phase_end

# ==========================
# redsea wrapper
# ==========================
def _run_redsea(redsea_exe: str, wav_path: str) -> Dict[str, Any]:
    """
    Run redsea in "json output" mode (best effort across versions).
    We try a small set of argument variants to handle schema drift.
    """
    variants = [
        [redsea_exe, "-f", wav_path, "-o", "json", "-u", "-p", "--time-from-start"],
        [redsea_exe, "-f", wav_path, "-o", "json", "-p"],
        [redsea_exe, "-f", wav_path, "-o", "json"],
        [redsea_exe, "-f", wav_path],
    ]

    best = None
    for cmd in variants:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        out, err = p.communicate()
        result = {
            "cmd": " ".join(cmd),
            "returncode": p.returncode,
            "stdout": out or "",
            "stderr": err or "",
            "stderr_tail": "\n".join(tail_lines(err or "", 40)),
        }
        # prefer successful returncode, but keep the one with most JSON lines as fallback
        json_lines = sum(1 for _ in (out or "").splitlines() if _.strip().startswith("{"))
        result["json_line_count"] = int(json_lines)
        if best is None:
            best = result
        else:
            if (result["returncode"] == 0 and best["returncode"] != 0) or (result["json_line_count"] > best["json_line_count"]):
                best = result
        if result["returncode"] == 0 and result["json_line_count"] > 0:
            best = result
            break
    assert best is not None
    groups = list(parse_json_lines(best["stdout"]))
    best["decoded_groups"] = len(groups)
    best["groups_preview"] = groups[:10]
    best["groups"] = groups
    return best

def _summarize_redsea_groups(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract common fields from redsea JSON output.
    Heuristic because redsea output schema varies.
    """
    pi = None
    ps = None
    rt = None
    pty = None

    def pick(obj: Dict[str, Any], *keys):
        for k in keys:
            if k in obj and obj[k] not in (None, "", []):
                return obj[k]
        return None

    for g in groups:
        if pi is None:
            pi = pick(g, "pi", "PI")
        if ps is None:
            ps = pick(g, "ps", "PS", "program_service", "program_service_name")
        if rt is None:
            rt = pick(g, "rt", "RT", "radiotext", "radio_text")
        if pty is None:
            pty = pick(g, "pty", "PTY")

        if isinstance(g.get("rds"), dict):
            r = g["rds"]
            if pi is None:
                pi = pick(r, "pi", "PI")
            if ps is None:
                ps = pick(r, "ps", "PS")
            if rt is None:
                rt = pick(r, "rt", "RT")
            if pty is None:
                pty = pick(r, "pty", "PTY")

    # normalize PI if int
    if isinstance(pi, int):
        pi = f"{pi:04X}"

    return {"pi": pi, "ps": ps, "rt": rt, "pty": pty}

# ==========================
# Main decode pipeline
# ==========================
def decode_rds_for_station(
    cfg: CaptureConfig,
    tools: ToolPaths,
    *,
    event_q=None,
    db_path: Optional[str] = None,
    power_db: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Capture IQ, produce MPX WAV, run redsea, return summary + diagnostics.

    This function never uses Streamlit APIs; safe to run in a worker thread.
    """
    t0 = time.time()

    # Resolve tools (accept absolute paths or PATH names)
    hackrf_transfer_exe = resolve_tool(tools.hackrf_transfer, ["hackrf_transfer", "hackrf_transfer.exe"])
    redsea_exe = resolve_tool(tools.redsea, ["redsea", "redsea.exe"])

    if not hackrf_transfer_exe:
        raise RuntimeError("hackrf_transfer not found (path or PATH).")
    if not redsea_exe:
        raise RuntimeError("redsea not found (path or PATH).")

    # Choose output WAV near DB if provided
    if db_path:
        wav_path = str(Path(db_path).with_suffix(".last_mpx.wav"))
        wav_rds_path = str(Path(db_path).with_suffix(".last_rds_bpf.wav"))
    else:
        wav_path = str(Path.cwd() / "fm_rds_guide.last_mpx.wav")
        wav_rds_path = str(Path.cwd() / "fm_rds_guide.last_rds_bpf.wav")

    emit_event(event_q, {
        "type": "debug",
        "stage": "station_start",
        "cfg": cfg.to_jsonable(),
        "power_db": power_db,
        "wav_path": wav_path,
    })

    # Capture to temp file
    with tempfile.TemporaryDirectory(prefix="fm_rds_v31_") as td:
        iq_path = os.path.join(td, "capture_iq_u8.raw")

        cap = _run_hackrf_transfer_to_file(
            hackrf_transfer_exe,
            center_freq_hz=int(cfg.center_freq_hz),
            sample_rate_hz=int(cfg.sample_rate_hz),
            lna_db=int(cfg.lna_db),
            vga_db=int(cfg.vga_db),
            amp=bool(cfg.amp),
            seconds=float(cfg.seconds),
            out_path=iq_path,
        )
        emit_event(event_q, {
            "type": "debug",
            "stage": "capture_done",
            "returncode": cap["returncode"],
            "stderr_tail": cap["stderr_tail"],
            "lna_applied": cap["lna_applied"],
            "vga_applied": cap["vga_applied"],
            "n_samples": cap["n_samples"],
            "bytes_expected": int(cap["n_samples"]) * 2,
            "bytes_actual": os.path.getsize(iq_path) if os.path.exists(iq_path) else 0,
        })

        if cap["returncode"] != 0:
            raise RuntimeError("hackrf_transfer failed.\n\n" + (cap["stderr_tail"] or "(no stderr)"))

        if (not os.path.exists(iq_path)) or os.path.getsize(iq_path) < 4096:
            raise RuntimeError("capture file is empty/small; HackRF likely returned no samples.\n\n" + (cap["stderr_tail"] or ""))

        # DSP pipeline: file stream -> NCO mix -> FIR+decim -> FM demod => MPX
        taps = fir_lowpass(numtaps=161, cutoff_hz=float(cfg.cutoff_hz), fs=float(cfg.sample_rate_hz))
        dec = FIRDecimator(taps=taps, decim=int(cfg.decim))
        agc = MovingRMS(alpha=0.02, value=0.0)

        phase = 0.0
        prev_iq = None
        total_iq = 0
        iq_rms_acc = 0.0
        iq_mean_i_acc = 0.0
        iq_mean_q_acc = 0.0
        blocks = 0

        mpx_chunks: List[np.ndarray] = []
        fs_out = float(cfg.sample_rate_hz) / float(cfg.decim)

        # We'll process in ~0.25s IQ chunks to keep convolution cost reasonable
        chunk_iq_complex = int(float(cfg.sample_rate_hz) * 0.25)
        chunk_bytes = max(16384, chunk_iq_complex * 2)
        # ensure even byte count
        chunk_bytes -= (chunk_bytes % 2)

        with open(iq_path, "rb") as f:
            while True:
                raw = f.read(chunk_bytes)
                if not raw:
                    break
                iq = _u8iq_bytes_to_c64(raw)
                if iq.size == 0:
                    break

                # basic IQ stats (coarse)
                i = np.real(iq).astype(np.float32, copy=False)
                q = np.imag(iq).astype(np.float32, copy=False)
                iq_mean_i_acc += float(np.mean(i))
                iq_mean_q_acc += float(np.mean(q))
                iq_rms_acc += float(np.sqrt(np.mean(i*i + q*q)))
                blocks += 1

                # NCO mix (offset tuning)
                iq, phase = _nco_mix_chunk(iq, fs=float(cfg.sample_rate_hz), f_hz=float(cfg.nco_freq_hz), phase0=phase)

                # decimate
                iq_d = dec.process(iq)
                if iq_d.size == 0:
                    continue

                # FM demod (stitch chunk boundary)
                y, prev_iq = fm_quadrature_demod(iq_d, prev=prev_iq)
                if y.size == 0:
                    continue

                # Gentle AGC on MPX (helps consistent PCM scaling)
                rms = agc.update(y)
                target = 0.35
                if rms > 1e-6:
                    y = (y / rms) * target

                mpx_chunks.append(y.astype(np.float32, copy=False))
                total_iq += int(iq.size)

        if not mpx_chunks:
            raise RuntimeError("No MPX produced. Something went wrong in DSP pipeline.")

        mpx = np.concatenate(mpx_chunks).astype(np.float32, copy=False)

        # Drop settle window to avoid transients crushing RDS
        settle_n = int(float(cfg.settle_s) * fs_out)
        if settle_n > 0 and mpx.size > settle_n + int(0.5 * fs_out):
            mpx = mpx[settle_n:]

        pilot_snr_db, rds_snr_db = estimate_pilot_and_rds_snr_db(mpx, fs_out)

        # Convert MPX to PCM16 and write WAV
        pcm, pcm_scale, clip_pct = float_to_int16_pcm(mpx, percentile=99.9)
        _write_wav_pcm16_mono(wav_path, pcm, int(round(fs_out)))

        # Optional: create an RDS bandpass-only WAV (for debugging)
        try:
            bw = float(cfg.rds_bandpass_bw_hz) / 2.0
            bp = fir_bandpass(int(cfg.rds_bandpass_taps), RDS_SUBCARRIER_HZ - bw, RDS_SUBCARRIER_HZ + bw, fs_out)
            # filter in float domain
            rds = np.convolve(mpx, bp.astype(np.float32), mode="same").astype(np.float32)
            pcm_rds, _, _ = float_to_int16_pcm(rds, percentile=99.9)
            _write_wav_pcm16_mono(wav_rds_path, pcm_rds, int(round(fs_out)))
        except Exception:
            # don't fail decode on debug output
            pass

        emit_event(event_q, {
            "type": "debug",
            "stage": "mpx_ready",
            "fs_out": fs_out,
            "n_samples": int(mpx.size),
            "duration_s": float(mpx.size) / fs_out,
            "pilot_snr_db": pilot_snr_db,
            "rds_snr_db": rds_snr_db,
            "clip_pct": clip_pct,
            "wav_path": wav_path,
        })

        # Run redsea on the chosen input WAV
        redsea_wav = wav_path if str(cfg.redsea_input).lower() != "rds_bandpass" else wav_rds_path
        red = _run_redsea(redsea_exe, redsea_wav)
        groups = red.get("groups") or []
        summary = _summarize_redsea_groups(groups)

        elapsed = time.time() - t0
        iq_rms = (iq_rms_acc / blocks) if blocks else None
        iq_dc_i = (iq_mean_i_acc / blocks) if blocks else None
        iq_dc_q = (iq_mean_q_acc / blocks) if blocks else None

        result = {
            "station_freq_hz": int(cfg.station_freq_hz),
            "center_freq_hz": int(cfg.center_freq_hz),
            "power_db": power_db,
            "n_groups": int(len(groups)),
            "pi": summary.get("pi"),
            "ps": summary.get("ps"),
            "pty": summary.get("pty"),
            "rt": summary.get("rt"),
            "pilot_snr_db": pilot_snr_db,
            "rds_snr_db": rds_snr_db,
            "clip_pct": float(clip_pct),
            "iq_rms": float(iq_rms) if iq_rms is not None else None,
            "iq_dc_i": float(iq_dc_i) if iq_dc_i is not None else None,
            "iq_dc_q": float(iq_dc_q) if iq_dc_q is not None else None,
            "agc_rms": float(agc.value),
            "elapsed_s": float(elapsed),
            "wav_path": wav_path,
            "wav_rds_path": wav_rds_path,
            "redsea": {
                "cmd": red.get("cmd"),
                "returncode": red.get("returncode"),
                "decoded_groups": red.get("decoded_groups"),
                "stderr_tail": red.get("stderr_tail"),
                "groups_preview": red.get("groups_preview"),
            },
            "cfg": cfg.to_jsonable(),
        }

        emit_event(event_q, {
            "type": "debug",
            "stage": "decode_done",
            "n_groups": int(len(groups)),
            "pi": summary.get("pi"),
            "ps": summary.get("ps"),
            "elapsed_s": float(elapsed),
        })

        return result

def _write_wav_pcm16_mono(path: str, pcm: np.ndarray, sample_rate: int) -> None:
    pcm = np.asarray(pcm, dtype=np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())

# ==========================
# Auto-tuning
# ==========================
def score_decode_result(r: Dict[str, Any]) -> float:
    """
    Heuristic score to compare decode attempts.

    Prioritize:
    - decoded groups (strong signal of success)
    - RDS SNR (if computed)
    - penalize clipping
    """
    n_groups = float(r.get("n_groups") or 0.0)
    rds_snr = r.get("rds_snr_db")
    clip = float(r.get("clip_pct") or 0.0)

    score = 0.0
    score += n_groups * 10.0
    if rds_snr is not None:
        score += float(rds_snr) * 0.7
    # small bonus if PS exists
    if r.get("ps"):
        score += 15.0
    # penalty for clipping
    score -= clip * 1.2
    return float(score)

def auto_tune_station(
    station_freq_mhz: float,
    tools: ToolPaths,
    *,
    event_q=None,
    base_cfg: Optional[CaptureConfig] = None,
    trials: Optional[List[Dict[str, Any]]] = None,
    power_db: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Try a small grid of parameters and return the best result.
    This is intentionally conservative (fast + robust), not exhaustive.
    """
    from Defaults import make_capture_cfg, CaptureConfig

    if base_cfg is None:
        base_cfg = make_capture_cfg(float(station_freq_mhz))

    # Default trial grid
    if trials is None:
        trials = []
        for sr in (2_052_000, 2_280_000):
            for decim in (12,):
                for amp in (False, True):
                    for lna in (24, 32, 40):
                        for vga in (10, 20, 30, 40):
                            trials.append({
                                "sample_rate_hz": sr,
                                "decim": decim,
                                "amp": amp,
                                "lna_db": lna,
                                "vga_db": vga,
                                "offset_hz": base_cfg.offset_hz,
                                "redsea_input": "full",
                            })

        # Add a few "rds_bandpass" attempts
        trials += [{
            "sample_rate_hz": 2_280_000,
            "decim": 12,
            "amp": False,
            "lna_db": 40,
            "vga_db": 20,
            "offset_hz": base_cfg.offset_hz,
            "redsea_input": "rds_bandpass",
        }]

    best = None
    best_score = -1e9
    results: List[Dict[str, Any]] = []

    emit_event(event_q, {"type": "info", "stage": "autotune_start", "n_trials": len(trials)})

    for idx, t in enumerate(trials, 1):
        cfg = make_capture_cfg(
            float(station_freq_mhz),
            offset_hz=int(t.get("offset_hz", base_cfg.offset_hz)),
            sample_rate_hz=int(t.get("sample_rate_hz", base_cfg.sample_rate_hz)),
            decim=int(t.get("decim", base_cfg.decim)),
            lna_db=int(t.get("lna_db", base_cfg.lna_db)),
            vga_db=int(t.get("vga_db", base_cfg.vga_db)),
            amp=bool(t.get("amp", base_cfg.amp)),
            seconds=float(base_cfg.seconds),
            redsea_input=str(t.get("redsea_input", base_cfg.redsea_input)),
        )
        emit_event(event_q, {"type": "info", "stage": "autotune_trial", "i": idx, "cfg": cfg.to_jsonable()})
        try:
            r = decode_rds_for_station(cfg, tools, event_q=event_q, db_path=None, power_db=power_db)
            s = score_decode_result(r)
            r["score"] = s
            results.append(r)
        except Exception as e:
            emit_event(event_q, {"type": "error", "stage": "autotune_trial_fail", "i": idx, "err": str(e)})
            continue

        if s > best_score:
            best_score = s
            best = r

        # Early exit if we have a strong win
        if (best is not None) and (best.get("n_groups") or 0) >= 50:
            break

    if best is None:
        raise RuntimeError("Auto-tune failed: all trials errored.")

    emit_event(event_q, {"type": "info", "stage": "autotune_done", "best_score": best_score, "best_cfg": best.get("cfg")})
    best["autotune"] = {"best_score": best_score, "n_trials": len(trials), "results_count": len(results)}
    return best