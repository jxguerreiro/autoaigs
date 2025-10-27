#!/usr/bin/env python3
import os, sys, time, heapq, json
from typing import List, Tuple, Dict, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ======================= ENV & CONFIG =======================
def _env_list(key: str) -> List[str]:
    return [x.strip() for x in (os.getenv(key, "") or "").split(",") if x.strip()]

# Required
INPUT_DRIVE_FOLDER_ID = os.getenv("INPUT_DRIVE_FOLDER_ID", "")   # staging/finals folder (source)
DEST_FOLDERS          = _env_list("DEST_FOLDERS")                # comma-separated phone folder IDs (targets)

# Behavior
ACTION          = (os.getenv("ACTION", "MOVE") or "MOVE").upper()    # MOVE | COPY | SHORTCUT
BATCH_PER_RUN   = int(os.getenv("BATCH_PER_RUN", "0") or 0)          # 0 = unlimited
SORT_BY         = os.getenv("SORT_BY", "modifiedTime")               # name|createdTime|modifiedTime
SORT_DIR        = os.getenv("SORT_DIR", "desc").lower()              # asc|desc
DRY_RUN         = os.getenv("DRY_RUN", "0") == "1"
WIPE_DESTS      = os.getenv("WIPE_DESTS", "0") == "1"                # default OFF for safety

# Race-safety / stability (avoid overlapping with uploads)
WAIT_FOR_STABILITY_SEC = int(os.getenv("WAIT_FOR_STABILITY_SEC", "15") or 15)  # window of no changes
STABILITY_TIMEOUT_SEC  = int(os.getenv("STABILITY_TIMEOUT_SEC", "300") or 300) # max wait
POLL_INTERVAL_SEC      = float(os.getenv("POLL_INTERVAL_SEC", "2.0") or 2.0)

# Filtering (must match your finals)
ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
MIN_SIZE_MB  = float(os.getenv("MIN_SIZE_MB", "1.0") or 1.0)  # skip tiny/partial files

# Auth (shared with pipeline)
GDRIVE_SA_JSON = os.getenv("GDRIVE_SA_JSON", "")
GDRIVE_SA_INFO = os.getenv("GDRIVE_SA_INFO", "")

# ===========================================================
def _log(m: str): print(m, flush=True)

def _drive():
    scopes = ["https://www.googleapis.com/auth/drive"]
    if GDRIVE_SA_INFO:
        info = json.loads(GDRIVE_SA_INFO)
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    elif GDRIVE_SA_JSON:
        creds = service_account.Credentials.from_service_account_file(GDRIVE_SA_JSON, scopes=scopes)
    else:
        _log("ERROR: set GDRIVE_SA_JSON or GDRIVE_SA_INFO"); sys.exit(1)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _list_folder_files(drive, folder_id: str, fields: str = "id,name,parents,mimeType,createdTime,modifiedTime,size") -> List[Dict]:
    q = f"'{folder_id}' in parents and trashed=false"
    out, token = [], None
    while True:
        resp = drive.files().list(
            q=q,
            fields=f"nextPageToken, files({fields})",
            pageToken=token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        out.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token: break
    return out

def _only_videos(files: List[Dict]) -> List[Dict]:
    def _ok(f: Dict) -> bool:
        name = (f.get("name") or "").lower()
        ext  = "." + name.rsplit(".", 1)[-1] if "." in name else ""
        if ext not in ALLOWED_EXTS:
            return False
        # size check (skip zero/partial)
        try:
            sz = float(f.get("size") or 0.0)
        except Exception:
            sz = 0.0
        return (sz >= MIN_SIZE_MB * 1024 * 1024)
    return [f for f in files if _ok(f)]

def _stable_snapshot(drive, folder_id: str) -> Tuple[int, Optional[str]]:
    items = _list_folder_files(drive, folder_id, fields="id,modifiedTime")
    cnt = len(items)
    # derive a quick “fingerprint” from last modifiedTime max
    latest = max((it.get("modifiedTime","") for it in items), default="")
    return cnt, latest

def _wait_for_stable_folder(drive, folder_id: str, window_sec: int, timeout_sec: int, poll: float):
    _log(f"[WAIT] Ensuring input is stable for {window_sec}s (timeout {timeout_sec}s)…")
    start = time.time()
    # Two consecutive identical snapshots separated by window_sec ⇒ stable
    last_cnt, last_mod = _stable_snapshot(drive, folder_id)
    last_time = time.time()
    while True:
        time.sleep(poll)
        cnt, latest = _stable_snapshot(drive, folder_id)
        now = time.time()
        if cnt == last_cnt and latest == last_mod and (now - last_time) >= window_sec:
            _log("[WAIT] Input folder is stable.")
            return
        if cnt != last_cnt or latest != last_mod:
            last_cnt, last_mod = cnt, latest
            last_time = now
        if now - start > timeout_sec:
            _log("[WAIT] Timeout reached; proceeding anyway.")
            return

def _trash_all_in_folder(drive, folder_id: str):
    items = _list_folder_files(drive, folder_id, fields="id,name")
    if not items:
        _log(f"[WIPE] {folder_id}: already empty")
        return
    for it in items:
        try:
            if DRY_RUN:
                _log(f"[DRY-RUN][WIPE] would trash '{it['name']}' in {folder_id}")
                continue
            drive.files().update(
                fileId=it["id"],
                body={"trashed": True},
                fields="id",
                supportsAllDrives=True
            ).execute()
            _log(f"[WIPE] trashed '{it['name']}' in {folder_id}")
        except HttpError as e:
            _log(f"[WIPE][ERROR] {folder_id}: {e}")

def _count_in_folder(drive, folder_id: str) -> int:
    return len(_list_folder_files(drive, folder_id, fields="id"))

def _move_file(drive, file_id: str, new_parent: str, remove_parent: Optional[str]):
    if DRY_RUN:
        _log(f"[DRY-RUN][MOVE] {file_id}: {remove_parent or ''} → {new_parent}")
        return
    kwargs = dict(
        fileId=file_id,
        addParents=new_parent,
        fields="id,parents",
        supportsAllDrives=True
    )
    if remove_parent:
        kwargs["removeParents"] = remove_parent
    drive.files().update(**kwargs).execute()

def _copy_file(drive, file_id: str, new_parent: str, name: str):
    if DRY_RUN:
        _log(f"[DRY-RUN][COPY] {name} ({file_id}) → {new_parent}")
        return {"id": f"dryrun-copy-{file_id}"}
    return drive.files().copy(
        fileId=file_id,
        body={"name": name, "parents": [new_parent]},
        fields="id,parents",
        supportsAllDrives=True
    ).execute()

def _shortcut_file(drive, file_id: str, new_parent: str, name: str):
    if DRY_RUN:
        _log(f"[DRY-RUN][SHORTCUT] {name} ({file_id}) → {new_parent}")
        return {"id": f"dryrun-shortcut-{file_id}"}
    body = {
        "name": name,
        "mimeType": "application/vnd.google-apps.shortcut",
        "parents": [new_parent],
        "shortcutDetails": {"targetId": file_id}
    }
    return drive.files().create(
        body=body,
        fields="id,parents",
        supportsAllDrives=True
    ).execute()

def _sort_files(files: List[Dict]) -> List[Dict]:
    key = SORT_BY
    reverse = (SORT_DIR != "asc")
    if key == "name":
        files.sort(key=lambda f: f.get("name","").lower(), reverse=reverse)
    else:
        files.sort(key=lambda f: f.get(key, ""), reverse=reverse)  # ISO times sort lexicographically
    return files

# =============================== MAIN ===============================
def main():
    if not INPUT_DRIVE_FOLDER_ID:
        _log("ERROR: set INPUT_DRIVE_FOLDER_ID"); sys.exit(1)
    if not DEST_FOLDERS:
        _log("ERROR: set DEST_FOLDERS (comma-separated Drive folder IDs)"); sys.exit(1)
    if ACTION not in ("MOVE","COPY","SHORTCUT"):
        _log("ERROR: ACTION must be MOVE|COPY|SHORTCUT"); sys.exit(1)
    if INPUT_DRIVE_FOLDER_ID in DEST_FOLDERS:
        _log("ERROR: INPUT_DRIVE_FOLDER_ID must not be one of DEST_FOLDERS"); sys.exit(1)

    d = _drive()

    # 0) Wait until uploads are done (stable folder)
    _wait_for_stable_folder(
        d, INPUT_DRIVE_FOLDER_ID,
        window_sec=WAIT_FOR_STABILITY_SEC,
        timeout_sec=STABILITY_TIMEOUT_SEC,
        poll=POLL_INTERVAL_SEC
    )

    # 1) Optional wipe of destination phone folders (OFF by default)
    if WIPE_DESTS:
        _log("[STEP] Wiping destination folders…")
        for fid in DEST_FOLDERS:
            _trash_all_in_folder(d, fid)
    else:
        _log("[STEP] Skipping destination wipe (WIPE_DESTS=0)")

    # 2) Load & filter source files from input Drive folder
    _log(f"[STEP] Listing input files from {INPUT_DRIVE_FOLDER_ID}…")
    files = _list_folder_files(d, INPUT_DRIVE_FOLDER_ID)
    files = _only_videos(files)
    if not files:
        _log("[OK] No eligible video files found in input folder.")
        return

    files = _sort_files(files)
    if BATCH_PER_RUN > 0:
        files = files[:BATCH_PER_RUN]
        _log(f"[INFO] Limiting to first {BATCH_PER_RUN} files.")

    # 3) Build min-heap for balanced distribution (count, folder_id)
    heap: List[Tuple[int, str]] = [(0 if DRY_RUN else _count_in_folder(d, fid), fid) for fid in DEST_FOLDERS]
    heapq.heapify(heap)

    # 4) Distribute
    moved = copied = shortcutted = 0
    for f in files:
        fname = f.get("name","(unnamed)")
        fid   = f.get("id","")
        parents = f.get("parents", []) or []

        cnt, dst = heapq.heappop(heap)
        try:
            if ACTION == "MOVE":
                remove_parent = INPUT_DRIVE_FOLDER_ID if INPUT_DRIVE_FOLDER_ID in parents else None
                _move_file(d, fid, new_parent=dst, remove_parent=remove_parent)
                moved += 1
                _log(f"[MOVE] {fname} → {dst}")
            elif ACTION == "COPY":
                _copy_file(d, fid, new_parent=dst, name=fname)
                copied += 1
                _log(f"[COPY] {fname} → {dst}")
            else:  # SHORTCUT
                _shortcut_file(d, fid, new_parent=dst, name=fname)
                shortcutted += 1
                _log(f"[SHORTCUT] {fname} → {dst}")

            cnt += 1
        except HttpError as e:
            _log(f"[ERROR] {fname}: {e}")
        finally:
            heapq.heappush(heap, (cnt, dst))

        time.sleep(0.03)  # gentle throttle

    _log(f"[DONE] ACTION={ACTION} | moved={moved} copied={copied} shortcuts={shortcutted} | DRY_RUN={int(DRY_RUN)}")

if __name__ == "__main__":
    main()
