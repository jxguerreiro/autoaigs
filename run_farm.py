#!/usr/bin/env python3
import os
import sys
import shlex
import time
import subprocess
from pathlib import Path

# ---------- tiny .env loader ----------
def load_env(dotenv_path: Path = Path(".env")):
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# ---------- logging ----------
_T0 = time.time()
def ts(): return time.strftime("%Y-%m-%d %H:%M:%S")
def rel(): return f"{time.time()-_T0:7.2f}s"
def log(msg): print(f"[{ts()}] {msg}", flush=True)
def log_rel(msg): print(f"[{rel()}] {msg}", flush=True)

# ---------- utils ----------
def have(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None

def run(cmd: str, cwd: Path | None = None) -> int:
    log_rel(f"RUN: {cmd}")
    p = subprocess.run(shlex.split(cmd), cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode

def detect_device() -> str:
    gpu = have("nvidia-smi") and subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    cudnn9 = False
    if have("ldconfig"):
        try:
            out = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
            cudnn9 = ("libcudnn_ops.so.9" in out)
        except Exception:
            pass
    # fallback simple paths
    if not cudnn9:
        for p in (
            "/usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9",
            "/usr/local/cuda/lib64/libcudnn_ops.so.9",
            f"{os.environ.get('CONDA_PREFIX','')}/lib/libcudnn_ops.so.9",
        ):
            if p and Path(p).exists():
                cudnn9 = True
                break
    device = "cuda" if (gpu and cudnn9) else "cpu"
    log(f"GPU available: {gpu} | cuDNN9 available: {cudnn9} | selected device: {device}")
    return device

def set_runtime_env(device: str):
    # Whisper/faster-whisper knobs that main.py reads
    os.environ.setdefault("USE_FASTER_WHISPER", "1")
    os.environ["WHISPER_DEVICE"] = device
    os.environ["FASTER_WHISPER_DEVICE"] = device
    os.environ["FASTER_WHISPER_COMPUTE_TYPE"] = "int8_float16" if device == "cuda" else "int8"

    # Optional audio clamp defaults (leave overridable by .env)
    os.environ.setdefault("MIN_SILENCE_SEC", "0.7")
    os.environ.setdefault("SILENCE_NOISE_DB", "-35")

    # CUDA selection passthrough
    cuda_device = os.environ.get("CUDA_DEVICE", "0")
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        conda = os.environ.get("CONDA_PREFIX", "")
        if conda:
            os.environ["LD_LIBRARY_PATH"] = f"{conda}/lib:{os.environ.get('LD_LIBRARY_PATH','')}"

# ---------- phases with timers ----------
timings = {
    "drive_sync": None,
    "render": None,
    "distribute": None,
}

def time_phase(name: str, fn, *args, **kwargs):
    t0 = time.time()
    log(f"==> START {name.upper()} at {ts()}")
    try:
        return fn(*args, **kwargs)
    finally:
        dt = time.time() - t0
        timings[name] = dt
        log(f"<== END   {name.upper()} in {dt:.2f}s")

def phase_drive_sync(py: str, script: str):
    if not Path(script).exists():
        log("Drive sync script not found; skipping.")
        return
    if not (os.getenv("SCRIPTS_DRIVE_FOLDER_ID") or os.getenv("GREENSCREEN_DRIVE_FOLDER_ID")):
        log("Drive IDs not set; skipping Drive sync.")
        return
    run(f"{py} {script}")

def phase_render(py: str, script: str):
    if not Path(script).exists():
        raise FileNotFoundError(f"{script} not found")
    run(f"{py} {script}")

def phase_distribute(py: str, script: str):
    if os.getenv("RUN_DISTRIBUTOR", "1") not in ("1", "true", "True", "YES", "yes"):
        log("Distributor disabled via RUN_DISTRIBUTOR; skipping.")
        return
    if not Path(script).exists():
        log("Distributor script not found; skipping.")
        return
    run(f"{py} {script}")

# ---------- CLI ----------
def parse_argv():
    import argparse
    ap = argparse.ArgumentParser(description="Orchestrate Drive sync → render → distribute with timers.")
    ap.add_argument("--py", default=os.environ.get("PY", "python3"), help="Python interpreter (default: python3)")
    ap.add_argument("--drive-sync", default=os.environ.get("DL_SCRIPT", "drive_sync.py"), help="Drive sync script path")
    ap.add_argument("--main", default=os.environ.get("MAIN_SCRIPT", "main.py"), help="Renderer main script path")
    ap.add_argument("--dist", default=os.environ.get("DIST_SCRIPT", "dist.py"), help="Distributor script path")
    ap.add_argument("--no-sync", action="store_true", help="Skip Drive sync phase")
    ap.add_argument("--no-render", action="store_true", help="Skip render phase")
    ap.add_argument("--no-dist", action="store_true", help="Skip distribution phase")
    return ap.parse_args()

def print_summary():
    total = sum(dt for dt in timings.values() if dt is not None)
    log("----- PHASE TIMING SUMMARY -----")
    for k in ("drive_sync", "render", "distribute"):
        v = timings[k]
        pretty = f"{v:.2f}s" if v is not None else "skipped"
        log(f"{k:>12}: {pretty}")
    log(f"{'total':>12}: {total:.2f}s")
    log("--------------------------------")

def main():
    load_env()
    args = parse_argv()

    device = detect_device()
    set_runtime_env(device)

    if not args.no_sync:
        time_phase("drive_sync", phase_drive_sync, args.py, args.drive_sync)
    else:
        log("Skipping Drive sync (flag).")

    if not args.no_render:
        time_phase("render", phase_render, args.py, args.main)
    else:
        log("Skipping render (flag).")

    if not args.no_dist:
        time_phase("distribute", phase_distribute, args.py, args.dist)
    else:
        log("Skipping distribution (flag).")

    print_summary()
    log("RUN FARM COMPLETE.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)
