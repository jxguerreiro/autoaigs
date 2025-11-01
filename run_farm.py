#!/usr/bin/env python3
"""
run_farm.py — Upload finals in FINAL_VIDS_DIR to Google Drive and iterate the pipeline.

Key features:
- Full Drive scope: https://www.googleapis.com/auth/drive
- supportsAllDrives=True on all Drive calls
- Resolves shortcuts, validates folder IDs
- Resumable upload with retries + progress
- Logs Service Account identity
- Iterative loop:
    download 1 file → move to PROCESSING (optional) → render
    → upload finals → move original to ARCHIVE (or back to SOURCE on failure)
    → clean
- De-dup: local skip-list of seen file IDs, plus optional processing folder.
"""

import os
import sys
import shlex
import time
import json
import shutil
import mimetypes
import subprocess
from pathlib import Path
from typing import Iterable, Optional, List, Tuple

# ---------- locations ----------
HERE = Path(__file__).resolve().parent
STATE_DIR = HERE / "_state"
STATE_DIR.mkdir(exist_ok=True)
SEEN_IDS_PATH = STATE_DIR / "seen_ids.json"

# ---------- tiny .env loader ----------
def load_env(dotenv_path: Path = HERE / ".env"):
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

# ---------- logging ----------
_T0 = time.time()
def ts() -> str: return time.strftime("%Y-%m-%d %H:%M:%S")
def rel() -> str: return f"{time.time()-_T0:7.2f}s"
def log(msg: str): print(f"[{ts()}] {msg}", flush=True)
def log_rel(msg: str): print(f"[{rel()}] {msg}", flush=True)

# ---------- utils ----------
def have(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None

def run(cmd: Iterable[str] | str, cwd: Optional[Path] = None, env: Optional[dict] = None) -> int:
    if isinstance(cmd, list):
        pretty = " ".join(shlex.quote(c) for c in cmd)
        argv = cmd
    else:
        pretty = cmd
        argv = shlex.split(cmd)
    log_rel(f"RUN: {pretty}  (cwd={cwd or Path.cwd()})")
    p = subprocess.run(argv, cwd=str(cwd) if cwd else None, env={**os.environ, **(env or {})})
    log_rel(f"EXIT: {p.returncode}")
    if p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode

# ---- interpreter resolver ----
def _resolve_python(preferred: str | None) -> str:
    from shutil import which
    cand = (preferred or "").strip()
    if cand:
        p = Path(cand)
        if not p.is_absolute():
            p = (HERE / cand)
        if p.exists():
            return str(p)
        if which(cand):
            return which(cand)
    venv_py = HERE / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable or (which("python3") or "python3")

def detect_device() -> str:
    gpu = have("nvidia-smi") and subprocess.run(
        ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0
    cudnn9 = False
    if have("ldconfig"):
        try:
            out = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
            cudnn9 = ("libcudnn_ops.so.9" in out)
        except Exception:
            pass
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
    os.environ.setdefault("USE_FASTER_WHISPER", "1")
    os.environ["WHISPER_DEVICE"] = device
    os.environ["FASTER_WHISPER_DEVICE"] = device
    os.environ["FASTER_WHISPER_COMPUTE_TYPE"] = "int8_float16" if device == "cuda" else "int8"
    os.environ.setdefault("MIN_SILENCE_SEC", "0.7")
    os.environ.setdefault("SILENCE_NOISE_DB", "-35")
    cuda_device = os.environ.get("CUDA_DEVICE", "0")
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        conda = os.environ.get("CONDA_PREFIX", "")
        if conda:
            os.environ["LD_LIBRARY_PATH"] = f"{conda}/lib:{os.environ.get('LD_LIBRARY_PATH','')}"

# ---------- file helpers ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_exts() -> List[str]:
    ex = os.getenv("VIDEO_EXTS", ".mp4,.mov,.mkv,.avi,.webm")
    return [s.strip().lower() for s in ex.split(",") if s.strip()]

def list_videos(root: Path) -> List[Path]:
    exts = tuple(parse_exts())
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts],
                  key=lambda p: p.stat().st_mtime)

def wipe_dir(p: Path):
    if p.exists():
        for child in p.iterdir():
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                else:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception as e:
                log(f"warn: failed to remove {child}: {e}")

def chmod_everything(path: Path, mode: int = 0o777):
    if not path.exists(): return
    try:
        os.chmod(path, mode)
    except Exception:
        pass
    if path.is_dir():
        for root, dirs, files in os.walk(path):
            for d in dirs:
                try: os.chmod(Path(root) / d, mode)
                except: pass
            for f in files:
                try: os.chmod(Path(root) / f, mode)
                except: pass

# ---------- local state (seen IDs) ----------
def _load_seen_ids() -> set[str]:
    if SEEN_IDS_PATH.exists():
        try:
            return set(json.loads(SEEN_IDS_PATH.read_text()))
        except Exception:
            return set()
    return set()

def _save_seen_ids(ids: set[str]):
    try:
        SEEN_IDS_PATH.write_text(json.dumps(sorted(list(ids)), indent=2))
    except Exception:
        pass

def _mark_seen(fid: str):
    ids = _load_seen_ids()
    if fid not in ids:
        ids.add(fid)
        _save_seen_ids(ids)

# ---------- Google Drive: auth / helpers ----------
def _sa_credentials():
    js_path = os.getenv("GDRIVE_SA_JSON", "").strip()
    js_inline = os.getenv("GDRIVE_SA_INFO", "").strip()
    if not js_path and not js_inline:
        raise RuntimeError("Service account credentials not set (GDRIVE_SA_JSON or GDRIVE_SA_INFO).")
    from google.oauth2 import service_account
    scopes = ["https://www.googleapis.com/auth/drive"]  # FULL scope
    if js_path:
        return service_account.Credentials.from_service_account_file(js_path, scopes=scopes)
    import json as _json
    return service_account.Credentials.from_service_account_info(_json.loads(js_inline), scopes=scopes)

def _drive_service():
    from googleapiclient.discovery import build
    creds = _sa_credentials()
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _drive_whoami(svc):
    me = svc.about().get(fields="user(emailAddress,displayName)").execute().get("user", {})
    log(f"Drive as: {me.get('displayName')} <{me.get('emailAddress')}>")

def _drive_get(svc, file_id: str, fields: str):
    return svc.files().get(fileId=file_id, fields=fields, supportsAllDrives=True).execute()

def _resolve_shortcut_if_needed(svc, fid: str) -> tuple[str, dict]:
    meta = _drive_get(svc, fid, fields="id,name,mimeType,shortcutDetails,driveId,parents")
    if meta.get("mimeType") == "application/vnd.google-apps.shortcut":
        tgt = meta.get("shortcutDetails", {}).get("targetId")
        if not tgt:
            raise RuntimeError("Shortcut has no targetId.")
        log(f"Drive: {fid} is a shortcut → target {tgt}")
        meta = _drive_get(svc, tgt, fields="id,name,mimeType,driveId,parents")
        return tgt, meta
    return fid, meta

def _ensure_folder_id(svc, fid: str, label: str) -> str:
    real, meta = _resolve_shortcut_if_needed(svc, fid)
    mt = meta.get("mimeType")
    if mt != "application/vnd.google-apps.folder":
        raise RuntimeError(f"{label}: ID is not a folder (mimeType={mt}).")
    where = "Shared Drive" if meta.get("driveId") else "My Drive"
    log(f"{label}: OK id={meta.get('id')} name={meta.get('name')} ({where})")
    return real

def _list_folder_files(service, folder_id: str, fields: str = "id,name,mimeType,parents,modifiedTime,size") -> List[dict]:
    out, token = [], None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields=f"nextPageToken, files({fields})",
            pageToken=token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        out.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return out

def _is_text_like(name: str, mime: str) -> bool:
    return name.lower().endswith(".txt") or (mime or "").startswith("text/")

def _is_video_like(name: str, mime: str) -> bool:
    ext = Path(name).suffix.lower()
    return ext in set(parse_exts()) or (mime or "").startswith("video/")

def download_scripts_once(service, folder_id: str, dest_dir: Path):
    if not folder_id: return
    files = _list_folder_files(service, folder_id)
    scripts = [f for f in files if _is_text_like(f.get("name",""), f.get("mimeType",""))]
    if not scripts:
        log("GDRIVE: no scripts found.")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in scripts:
        name = f["name"]
        tgt = dest_dir / name
        if tgt.exists():
            continue
        _download_file(service, f["id"], tgt)
    log("GDRIVE: scripts synced once.")

def _download_file(service, file_id: str, dest_path: Path, chunksize_mb: int = 8):
    from googleapiclient.http import MediaIoBaseDownload
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dest_path, "wb") as f:
        dl = MediaIoBaseDownload(f, req, chunksize=chunksize_mb*1024*1024)
        done = False
        while not done:
            _, done = dl.next_chunk()

def _move_file_between_folders(service, file_id: str, add_parent: str, remove_parents: List[str] | None):
    kwargs = dict(
        fileId=file_id,
        addParents=add_parent,
        fields="id,parents",
        supportsAllDrives=True
    )
    if remove_parents:
        kwargs["removeParents"] = ",".join(remove_parents)
    service.files().update(**kwargs).execute()

def _file_exists(service, file_id: str) -> bool:
    try:
        service.files().get(fileId=file_id, fields="id", supportsAllDrives=True).execute()
        return True
    except Exception:
        return False

def pick_next_video_meta(service, source_folder_id: str, seen_ids: set[str]) -> Optional[dict]:
    """Pick oldest video NOT in seen_ids."""
    items = _list_folder_files(service, source_folder_id, fields="id,name,mimeType,parents,modifiedTime,size")
    vids = [f for f in items if _is_video_like(f.get("name",""), f.get("mimeType",""))]
    if not vids:
        return None
    vids.sort(key=lambda f: f.get("modifiedTime",""))
    for f in vids:
        if f["id"] not in seen_ids:
            return f
    # if all seen, return None
    return None

def download_one_video(service, meta: dict, dest_dir: Path) -> Path:
    """Download specific meta to dest; returns local Path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = meta["name"]
    local = dest_dir / name
    base = local; i = 1
    while local.exists():
        local = base.with_name(f"{base.stem}-{i}{base.suffix}")
        i += 1
    _download_file(service, meta["id"], local)
    log(f"GDRIVE: downloaded video {local.name}")
    return local

def _resumable_upload(svc, parent_folder_id: str, file_path: Path, max_retries: int = 5) -> str:
    from googleapiclient.http import MediaFileUpload
    import random
    mime, _ = mimetypes.guess_type(str(file_path))
    media = MediaFileUpload(str(file_path), mimetype=mime or "application/octet-stream", resumable=True)
    req = svc.files().create(
        body={"name": file_path.name, "parents": [parent_folder_id]},
        media_body=media,
        fields="id,name,parents",
        supportsAllDrives=True
    )
    retry, response = 0, None
    while response is None:
        try:
            status, response = req.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                log(f"[upload] {file_path.name}: {pct}%")
        except Exception as e:
            retry += 1
            if retry > max_retries:
                raise
            sleep = min(30, (2 ** retry) + random.random())
            log(f"[upload] chunk error (retry {retry}/{max_retries} in {sleep:.1f}s): {e}")
            time.sleep(sleep)
    return response["id"]

def upload_to_drive(file_path: Path, parent_folder_id: str) -> str:
    svc = _drive_service()
    real_parent = _ensure_folder_id(svc, parent_folder_id, "UPLOAD_PARENT")
    log(f"Uploading {file_path.name} → Drive folder {real_parent} ...")
    fid = _resumable_upload(svc, real_parent, file_path)
    log(f"Uploaded {file_path.name} (id={fid})")
    return fid

def move_drive_file_to_archive(file_id: str, archive_folder_id: str, remove_parents: List[str] | None):
    svc = _drive_service()
    real_archive = _ensure_folder_id(svc, archive_folder_id, "ARCHIVE_PARENT")
    kwargs = dict(
        fileId=file_id,
        addParents=real_archive,
        fields="id,parents",
        supportsAllDrives=True
    )
    if remove_parents:
        kwargs["removeParents"] = ",".join(remove_parents)
    svc.files().update(**kwargs).execute()

def find_by_name_in_folder(svc, folder_id: str, name: str) -> List[dict]:
    safe = name.replace("'", "\\'")
    q = f"name = '{safe}' and '{folder_id}' in parents and trashed = false"
    resp = svc.files().list(
        q=q,
        fields="files(id,parents,name,mimeType,size,modifiedTime)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    return resp.get("files", [])

# ---------- phases ----------
def phase_render_single(py: str, script: str, single_src_dir: Path, env_overrides: dict | None = None):
    sp = Path(script)
    log(f"[render] single-file run | script={sp} (abs={sp.resolve()}) exists={sp.exists()}")
    if not sp.exists():
        raise FileNotFoundError(f"{script} not found")
    env = dict(env_overrides or {})
    env["GREENSCREEN_DIR"] = str(single_src_dir)
    run([py, str(sp)], cwd=HERE, env=env)

def phase_distribute(py: str, script: str):
    guard = os.getenv("RUN_DISTRIBUTOR", "1")
    sp = Path(script)
    log(f"[distribute] RUN_DISTRIBUTOR={guard} | script={sp} (abs={sp.resolve()}) exists={sp.exists()}")
    if guard.lower() not in ("1", "true", "yes", "y"):
        log("Distributor disabled via RUN_DISTRIBUTOR; skipping.")
        return
    if not sp.exists():
        log("Distributor script not found; skipping.")
        return
    run([py, str(sp)], cwd=HERE)

# ---------- CLI ----------
def parse_argv():
    import argparse
    ap = argparse.ArgumentParser(description="Iterative pipeline: scripts-once → (per video: dl→render→upload finals→archive original) → dist at end.")
    ap.add_argument("--py", default=os.environ.get("PY"), help="Python interpreter (defaults to .venv/bin/python if exists, else current)")
    ap.add_argument("--main", default=os.environ.get("MAIN_SCRIPT", str(HERE / "main.py")), help="Renderer main script path")
    ap.add_argument("--dist", default=os.environ.get("DIST_SCRIPT", str(HERE / "dist.py")), help="Distributor script path")
    ap.add_argument("--single-tmp", default=str(HERE / "_single_run"), help="Temp folder to isolate single-file runs")
    ap.add_argument("--max-empty-syncs", type=int, default=int(os.getenv("MAX_EMPTY_SYNCS", "2")), help="Consecutive empty syncs before stop")
    return ap.parse_args()

# ---------- main loop ----------
def main():
    load_env()
    args = parse_argv()
    args.py = _resolve_python(args.py)

    log(f"Using Python interpreter: {args.py}")

    device = detect_device()
    set_runtime_env(device)

    # Paths (local working dirs)
    GREENSCREEN_DIR = Path(os.getenv("GREENSCREEN_DIR", "./greenscreen")).resolve()
    SCRIPTS_DIR     = Path(os.getenv("SCRIPTS_DIR", "./scripts")).resolve()
    OUT_DIR         = Path(os.getenv("OUT_DIR", "./output_split")).resolve()
    FINAL_VIDS_DIR  = Path(os.getenv("FINAL_VIDS_DIR", "./final_videos")).resolve()
    ensure_dir(GREENSCREEN_DIR); ensure_dir(SCRIPTS_DIR); ensure_dir(OUT_DIR); ensure_dir(FINAL_VIDS_DIR)

    # Drive IDs
    GREENSCREEN_DRIVE_FOLDER_ID = os.getenv("GREENSCREEN_DRIVE_FOLDER_ID", "").strip()
    SCRIPTS_DRIVE_FOLDER_ID     = os.getenv("SCRIPTS_DRIVE_FOLDER_ID", "").strip()
    DRIVE_UPLOAD_FOLDER_ID      = os.getenv("DRIVE_UPLOAD_FOLDER_ID", "").strip()
    ARCHIVE_DRIVE_FOLDER_ID     = os.getenv("ARCHIVE_DRIVE_FOLDER_ID", "").strip()
    PROCESSING_DRIVE_FOLDER_ID  = os.getenv("PROCESSING_DRIVE_FOLDER_ID", "").strip()  # optional

    if not GREENSCREEN_DRIVE_FOLDER_ID:
        raise SystemExit("GREENSCREEN_DRIVE_FOLDER_ID not set.")
    if not DRIVE_UPLOAD_FOLDER_ID:
        raise SystemExit("DRIVE_UPLOAD_FOLDER_ID not set.")
    if not ARCHIVE_DRIVE_FOLDER_ID:
        raise SystemExit("ARCHIVE_DRIVE_FOLDER_ID not set.")

    # Behavior toggles
    DELETE_LOCAL_FINALS = os.getenv("DELETE_LOCAL_FINALS", "1").lower() in ("1", "true", "yes", "y")
    SAFETY_REQUIRE_UPLOAD = os.getenv("SAFETY_REQUIRE_UPLOAD", "1").lower() in ("1", "true", "yes", "y")

    # Single-file staging directory
    single_tmp_root = Path(args.single_tmp).resolve()
    single_src_dir  = single_tmp_root / "greenscreen_single"
    ensure_dir(single_src_dir)

    # Drive client + whoami + resolve folder IDs
    svc = _drive_service()
    _drive_whoami(svc)
    GREENSCREEN_DRIVE_FOLDER_ID = _ensure_folder_id(svc, GREENSCREEN_DRIVE_FOLDER_ID, "GREENSCREEN_PARENT")
    DRIVE_UPLOAD_FOLDER_ID      = _ensure_folder_id(svc, DRIVE_UPLOAD_FOLDER_ID, "UPLOAD_PARENT")
    ARCHIVE_DRIVE_FOLDER_ID     = _ensure_folder_id(svc, ARCHIVE_DRIVE_FOLDER_ID, "ARCHIVE_PARENT")
    if PROCESSING_DRIVE_FOLDER_ID:
        PROCESSING_DRIVE_FOLDER_ID = _ensure_folder_id(svc, PROCESSING_DRIVE_FOLDER_ID, "PROCESSING_PARENT")
    if SCRIPTS_DRIVE_FOLDER_ID:
        SCRIPTS_DRIVE_FOLDER_ID = _ensure_folder_id(svc, SCRIPTS_DRIVE_FOLDER_ID, "SCRIPTS_PARENT")

    # Scripts once
    if SCRIPTS_DRIVE_FOLDER_ID:
        download_scripts_once(svc, SCRIPTS_DRIVE_FOLDER_ID, SCRIPTS_DIR)

    log("===== START ITERATIVE MODE =====")
    empty_syncs = 0
    processed = 0
    seen_ids = _load_seen_ids()

    while True:
        vids = list_videos(GREENSCREEN_DIR)
        drive_meta = None

        if not vids:
            # pick next meta that isn't seen
            meta = pick_next_video_meta(svc, GREENSCREEN_DRIVE_FOLDER_ID, seen_ids)
            if not meta:
                empty_syncs += 1
                log(f"Drive empty or all seen. empty_syncs={empty_syncs}")
                if empty_syncs >= int(os.getenv("MAX_EMPTY_SYNCS", "2")):
                    break
                time.sleep(3)
                continue

            fid = meta["id"]
            parents = meta.get("parents", []) or []

            # Immediately move to PROCESSING (if configured) to avoid re-pick
            if PROCESSING_DRIVE_FOLDER_ID:
                try:
                    _move_file_between_folders(svc, fid, PROCESSING_DRIVE_FOLDER_ID, parents)
                    log(f"Moved source to PROCESSING (id={fid}).")
                    # Update parents to new context
                    parents = [PROCESSING_DRIVE_FOLDER_ID]
                except Exception as e:
                    log(f"warn: could not move to PROCESSING (id={fid}): {e}")

            local_path = download_one_video(svc, meta, GREENSCREEN_DIR)
            vids = [local_path]
            drive_meta = (fid, parents)
            _mark_seen(fid)  # remember this file id so we never re-pick it this run
            seen_ids.add(fid)
        else:
            # local file (no Drive meta known)
            drive_meta = (None, None)

        empty_syncs = 0
        src = vids[0]
        log(f"Picked: {src.name}")

        # Stage single file
        chmod_everything(single_src_dir, 0o777)
        for child in single_src_dir.iterdir():
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                else:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                pass

        staged = single_src_dir / src.name
        try:
            src.rename(staged)
        except Exception:
            shutil.move(str(src), str(staged))

        # 1) Render
        try:
            phase_render_single(args.py, args.main, single_src_dir, env_overrides={})
        except Exception as e:
            log(f"RENDER ERROR: {e}")
            # Move Drive file back to SOURCE if we had moved it to PROCESSING
            if drive_meta and drive_meta[0] and PROCESSING_DRIVE_FOLDER_ID:
                fid, parents = drive_meta
                try:
                    _move_file_between_folders(svc, fid, GREENSCREEN_DRIVE_FOLDER_ID, [PROCESSING_DRIVE_FOLDER_ID])
                    log(f"Returned file to SOURCE (id={fid}) after render error.")
                except Exception as e2:
                    log(f"warn: could not return file to SOURCE: {e2}")
            raise

        # 2) Upload finals found in FINAL_VIDS_DIR (mtime within window)
        uploaded_any = False
        now = time.time()
        window_sec = int(os.getenv("UPLOAD_WINDOW_SEC", "900"))
        finals = [p for p in FINAL_VIDS_DIR.glob("*") if p.is_file()]
        finals.sort(key=lambda p: p.stat().st_mtime)
        for f in finals:
            if (now - f.stat().st_mtime) <= window_sec:
                try:
                    upload_to_drive(f, DRIVE_UPLOAD_FOLDER_ID)
                    uploaded_any = True
                    if DELETE_LOCAL_FINALS:
                        try:
                            f.unlink(missing_ok=True)
                        except Exception as e:
                            log(f"warn: could not delete final {f.name}: {e}")
                except Exception as e:
                    log(f"ERROR uploading final {f.name}: {e}")

        # 3) Move original on Drive appropriately
        try:
            if SAFETY_REQUIRE_UPLOAD and not uploaded_any:
                log("SAFETY: No finals uploaded — keeping original in its current Drive folder.")
                # If we had moved it to PROCESSING, return it to SOURCE to retry later
                if drive_meta and drive_meta[0] and PROCESSING_DRIVE_FOLDER_ID:
                    fid, parents = drive_meta
                    # Only move back if currently in PROCESSING and not already gone
                    if _file_exists(svc, fid):
                        try:
                            _move_file_between_folders(svc, fid, GREENSCREEN_DRIVE_FOLDER_ID, [PROCESSING_DRIVE_FOLDER_ID])
                            log(f"Returned file to SOURCE (id={fid}) because no finals uploaded.")
                        except Exception as e2:
                            log(f"warn: could not return file to SOURCE: {e2}")
            else:
                # Success path: archive
                if drive_meta and drive_meta[0]:
                    fid, parents = drive_meta
                    # remove whichever parents it currently has (PROCESSING or SOURCE)
                    rem = parents or []
                    try:
                        _move_file_between_folders(svc, fid, ARCHIVE_DRIVE_FOLDER_ID, rem)
                        log(f"Moved original to ARCHIVE (id={fid}).")
                    except Exception as e:
                        log(f"ERROR moving original to ARCHIVE: {e}")
                else:
                    # Fallback: find by name in source folder (rare if we used PROCESSING)
                    files = find_by_name_in_folder(svc, GREENSCREEN_DRIVE_FOLDER_ID, staged.name)
                    if files:
                        fid = files[0]["id"]
                        parents = files[0].get("parents", []) or []
                        try:
                            _move_file_between_folders(svc, fid, ARCHIVE_DRIVE_FOLDER_ID, parents)
                            log(f"Moved (name-match) original to ARCHIVE (id={fid}).")
                        except Exception as e:
                            log(f"ERROR moving (name-match) to ARCHIVE: {e}")
                    else:
                        # Truly not found — upload the source to ARCHIVE to keep a record
                        log("Original not found in SOURCE; uploading staging copy to ARCHIVE as fallback.")
                        try:
                            upload_to_drive(staged, ARCHIVE_DRIVE_FOLDER_ID)
                        except Exception as e:
                            log(f"ERROR uploading source to ARCHIVE: {e}")
        finally:
            # 4) Remove local staged source
            try:
                os.chmod(staged, 0o666)
            except Exception:
                pass
            try:
                staged.unlink(missing_ok=True)
            except Exception as e:
                log(f"warn: could not delete local source: {e}")

            # 5) Cleanup intermediates aggressively
            chmod_everything(OUT_DIR, 0o777)
            wipe_dir(OUT_DIR)
            if DELETE_LOCAL_FINALS:
                chmod_everything(FINAL_VIDS_DIR, 0o777)
                wipe_dir(FINAL_VIDS_DIR)

        processed += 1
        log(f"Iteration complete. Processed so far: {processed}")

    # After all files are processed & uploaded, run distributor ONCE
    log("===== ALL FILES DONE → running distributor once =====")
    phase_distribute(args.py, args.dist)

    log(f"RUN COMPLETE. Total processed: {processed}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)
