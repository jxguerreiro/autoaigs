#!/usr/bin/env python3
import os, sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================== timers / logging ===============================
_T0 = time.time()
def _rel(): return time.time() - _T0
def log(msg: str): print(f"[{_rel():7.2f}s] {msg}", flush=True)

# =============================== tiny .env loader ===============================
def _load_env(dotenv_path: Path = Path(".env")):
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()

# =============================== env & paths ===============================
def _pp(key: str, default: str) -> Path:
    return Path(os.getenv(key, default)).expanduser().resolve()

# Local cache dirs (created if missing)
ROOT            = Path(__file__).resolve().parent
GREENSCREEN_DIR = _pp("GREENSCREEN_DIR", str(ROOT / "greenscreen"))
SCRIPTS_DIR     = _pp("SCRIPTS_DIR",     str(ROOT / "scripts"))

# Drive folder ids
SCRIPTS_DRIVE_FOLDER_ID     = os.getenv("SCRIPTS_DRIVE_FOLDER_ID", "")
GREENSCREEN_DRIVE_FOLDER_ID = os.getenv("GREENSCREEN_DRIVE_FOLDER_ID", "")

# Behavior
VIDEO_DOWNLOAD_WORKERS   = int(os.getenv("VIDEO_DOWNLOAD_WORKERS", "2") or 2)
DRIVE_RECURSE_SUBFOLDERS = os.getenv("DRIVE_RECURSE_SUBFOLDERS", "1") == "1"

# Auth
GDRIVE_SA_JSON = os.getenv("GDRIVE_SA_JSON", "")
GDRIVE_SA_INFO = os.getenv("GDRIVE_SA_INFO", "")

# =============================== Drive client ===============================
def _drive_scopes():
    return ["https://www.googleapis.com/auth/drive"]

def build_drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    if GDRIVE_SA_INFO:
        info  = json.loads(GDRIVE_SA_INFO)
        creds = service_account.Credentials.from_service_account_info(info, scopes=_drive_scopes())
    elif GDRIVE_SA_JSON:
        creds = service_account.Credentials.from_service_account_file(GDRIVE_SA_JSON, scopes=_drive_scopes())
    else:
        raise RuntimeError("Google Drive auth missing. Set GDRIVE_SA_JSON or GDRIVE_SA_INFO")
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# =============================== Drive utils ===============================
def _list_folder_files(service, folder_id: str):
    files, token = [], None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields=("nextPageToken, files("
                    "id,name,mimeType,size,modifiedTime,parents,shortcutDetails)"),
            pageToken=token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000
        ).execute()
        files.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return files

_VALID_VID_EXTS = {".mp4",".mov",".mkv",".webm",".m4v"}
def _is_text_like(name: str, mime: str):
    return name.lower().endswith(".txt") or (mime or "").startswith("text/")

def _is_video_like(name: str, mime: str):
    return Path(name).suffix.lower() in _VALID_VID_EXTS or (mime or "").startswith("video/")

def _unique_dest(base: Path) -> Path:
    if not base.exists():
        return base
    stem, ext = base.stem, base.suffix
    i = 1
    while True:
        cand = base.with_name(f"{stem}-{i}{ext}")
        if not cand.exists():
            return cand
        i += 1

def _download_file(service, file_id: str, dest_path: Path, chunksize_mb: int = 8):
    from googleapiclient.http import MediaIoBaseDownload
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dest_path, "wb") as f:
        dl = MediaIoBaseDownload(f, req, chunksize=chunksize_mb*1024*1024)
        done = False
        while not done:
            _, done = dl.next_chunk()

# =============================== Public APIs ===============================
def download_scripts_first(service, folder_id: str, dest_dir: Path):
    if not folder_id:
        return
    log("GDRIVE: listing scripts...")
    files   = _list_folder_files(service, folder_id)
    scripts = [f for f in files if _is_text_like(f["name"], f.get("mimeType",""))]
    if not scripts:
        log("GDRIVE: no scripts found in folder.")
        return
    for f in scripts:
        name = f["name"]
        dest = _unique_dest(dest_dir / name)
        _download_file(service, f["id"], dest)
        log(f"GDRIVE: downloaded script {dest.name}")

def _walk(service, fid: str, recurse: bool):
    for f in _list_folder_files(service, fid):
        mime = f.get("mimeType","")
        if mime == "application/vnd.google-apps.folder" and recurse:
            yield from _walk(service, f["id"], recurse)
        else:
            yield f

def download_all_videos(service, folder_id: str, dest_dir: Path, workers: int = 2, recurse: bool = True):
    if not folder_id:
        return []

    items = list(_walk(service, folder_id, recurse))
    jobs  = []
    for f in items:
        name = f.get("name", "unnamed")
        mime = f.get("mimeType","")
        if mime == "application/vnd.google-apps.shortcut":
            det      = f.get("shortcutDetails") or {}
            tgt_id   = det.get("targetId") or ""
            tgt_mime = det.get("targetMimeType") or ""
            if tgt_id and _is_video_like(name, tgt_mime):
                jobs.append({"file_id": tgt_id, "name": name})
        elif _is_video_like(name, mime):
            jobs.append({"file_id": f["id"], "name": name})

    if not jobs:
        log("GDRIVE: no videos found in greenscreen folder.")
        return []

    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"GDRIVE: downloading ALL greenscreen videos… (jobs={len(jobs)}, workers={max(1,workers)})")

    def _one(job):
        # Short-lived client per thread avoids transport reuse issues.
        svc   = build_drive_service()
        local = _unique_dest(dest_dir / job["name"])
        _download_file(svc, job["file_id"], local)
        return local

    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                p = fut.result()
                log(f"GDRIVE: downloaded video {p.name}")
                results.append(p)
            except Exception as e:
                log(f"GDRIVE: failed video download: {e}")
    return results

# =============================== CLI main ===============================
def main():
    # Ensure local dirs
    GREENSCREEN_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build client
    svc = build_drive_service()

    # Phase A1: scripts
    if SCRIPTS_DRIVE_FOLDER_ID:
        download_scripts_first(svc, SCRIPTS_DRIVE_FOLDER_ID, SCRIPTS_DIR)
        log("GDRIVE: scripts synced.")
    else:
        log("GDRIVE: no SCRIPTS_DRIVE_FOLDER_ID set; skipping scripts.")

    # Phase A2: videos
    if GREENSCREEN_DRIVE_FOLDER_ID:
        download_all_videos(
            service=svc,
            folder_id=GREENSCREEN_DRIVE_FOLDER_ID,
            dest_dir=GREENSCREEN_DIR,
            workers=VIDEO_DOWNLOAD_WORKERS,
            recurse=DRIVE_RECURSE_SUBFOLDERS
        )
    else:
        log("GDRIVE: no GREENSCREEN_DRIVE_FOLDER_ID set; skipping video downloads.")

    # Done
    log("DOWNLOAD PHASE COMPLETE.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)
