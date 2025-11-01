#!/usr/bin/env python3
import os, sys, time, heapq, json
from typing import List, Tuple, Dict, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ======================= ENV & CONFIG =======================
def _env_list(key: str) -> List[str]:
    raw = os.getenv(key, "") or ""
    return [x.strip() for x in raw.split(",") if x.strip()]

# Required (provided by pipeline env)
INPUT_DRIVE_FOLDER_ID = os.getenv("INPUT_DRIVE_FOLDER_ID", "")
DEST_FOLDERS_TT       = _env_list("DEST_FOLDERS_TT")
DEST_FOLDERS_IG       = _env_list("DEST_FOLDERS_IG")

# Behavior (from env; no .env loading)
BATCH_PER_RUN   = int(os.getenv("BATCH_PER_RUN", "0") or 0)      # 0 = unlimited
SORT_BY         = os.getenv("SORT_BY", "modifiedTime")
SORT_DIR        = os.getenv("SORT_DIR", "desc").lower()
DRY_RUN         = os.getenv("DRY_RUN", "0") == "1"

# Always wipe destinations before distributing (Trash)
WIPE_DESTS = True

# Race-safety / stability
WAIT_FOR_STABILITY_SEC = int(os.getenv("WAIT_FOR_STABILITY_SEC", "15") or 15)
STABILITY_TIMEOUT_SEC  = int(os.getenv("STABILITY_TIMEOUT_SEC", "300") or 300)
POLL_INTERVAL_SEC      = float(os.getenv("POLL_INTERVAL_SEC", "2.0") or 2.0)

# Filtering
ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
MIN_SIZE_MB  = float(os.getenv("MIN_SIZE_MB", "1.0") or 1.0)

# Auth
GDRIVE_SA_JSON = os.getenv("GDRIVE_SA_JSON", "")
GDRIVE_SA_INFO = os.getenv("GDRIVE_SA_INFO", "")

# ===========================================================
def _log(m: str): print(m, flush=True)

def _drive():
    scopes = ["https://www.googleapis.com/auth/drive"]
    try:
        if GDRIVE_SA_INFO:
            info = json.loads(GDRIVE_SA_INFO)
            creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        elif GDRIVE_SA_JSON:
            creds = service_account.Credentials.from_service_account_file(GDRIVE_SA_JSON, scopes=scopes)
        else:
            _log("ERROR: set GDRIVE_SA_JSON or GDRIVE_SA_INFO"); sys.exit(1)
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        _log(f"ERROR: auth/build drive client failed: {e}")
        sys.exit(1)

def _resolve_shortcut(drive, fid: str) -> Tuple[str, Dict]:
    meta = drive.files().get(
        fileId=fid,
        fields="id,name,mimeType,shortcutDetails,parents,driveId",
        supportsAllDrives=True
    ).execute()
    if meta.get("mimeType") == "application/vnd.google-apps.shortcut":
        tgt = meta.get("shortcutDetails", {}).get("targetId")
        if not tgt:
            raise RuntimeError(f"Shortcut {fid} has no targetId")
        meta = drive.files().get(
            fileId=tgt,
            fields="id,name,mimeType,parents,driveId",
            supportsAllDrives=True
        ).execute()
        return meta["id"], meta
    return meta["id"], meta

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
        try:
            sz = float(f.get("size") or 0.0)
        except Exception:
            sz = 0.0
        return (sz >= MIN_SIZE_MB * 1024 * 1024)
    return [f for f in files if _ok(f)]

def _stable_snapshot(drive, folder_id: str) -> Tuple[int, Optional[str]]:
    items = _list_folder_files(drive, folder_id, fields="id,modifiedTime")
    cnt = len(items)
    latest = max((it.get("modifiedTime","") for it in items), default="")
    return cnt, latest

def _wait_for_stable_folder(drive, folder_id: str, window_sec: int, timeout_sec: int, poll: float):
    _log(f"[WAIT] Ensuring input is stable for {window_sec}s (timeout {timeout_sec}s)…")
    start = time.time()
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

def _trash_all_in_folder(drive, folder_id: str) -> int:
    items = _list_folder_files(drive, folder_id, fields="id,name")
    if not items:
        _log(f"[WIPE] {folder_id}: already empty")
        return 0
    trashed = 0
    for it in items:
        try:
            if DRY_RUN:
                _log(f"[DRY-RUN][WIPE] would trash '{it['name']}' in {folder_id}")
                trashed += 1
                continue
            drive.files().update(
                fileId=it["id"],
                body={"trashed": True},
                fields="id",
                supportsAllDrives=True
            ).execute()
            trashed += 1
        except HttpError as e:
            _log(f"[WIPE][ERROR] {folder_id}: {e}")
    _log(f"[WIPE] {folder_id}: trashed {trashed} item(s)")
    return trashed

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

def _shortcut_file(drive, target_id: str, new_parent: str, name: str):
    if DRY_RUN:
        _log(f"[DRY-RUN][SHORTCUT] {name} ({target_id}) → {new_parent}")
        return {"id": f"dryrun-shortcut-{target_id}"}
    body = {
        "name": name,
        "mimeType": "application/vnd.google-apps.shortcut",
        "parents": [new_parent],
        "shortcutDetails": {"targetId": target_id}
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
        files.sort(key=lambda f: f.get(key, ""), reverse=reverse)
    return files

# =============================== MAIN ===============================
def _print_config():
    _log("[CONFIG] BATCH_PER_RUN=%s SORT_BY=%s SORT_DIR=%s DRY_RUN=%s WIPE_DESTS=%s" %
         (BATCH_PER_RUN, SORT_BY, SORT_DIR, int(DRY_RUN), int(WIPE_DESTS)))
    _log("[CONFIG] INPUT_DRIVE_FOLDER_ID=%s" % INPUT_DRIVE_FOLDER_ID)
    _log("[CONFIG] TT(count=%d)=%s" % (len(DEST_FOLDERS_TT), ",".join(DEST_FOLDERS_TT)))
    _log("[CONFIG] IG(count=%d)=%s" % (len(DEST_FOLDERS_IG), ",".join(DEST_FOLDERS_IG)))

def main():
    if not INPUT_DRIVE_FOLDER_ID:
        _log("ERROR: set INPUT_DRIVE_FOLDER_ID"); sys.exit(1)
    if not DEST_FOLDERS_TT or not DEST_FOLDERS_IG:
        _log("ERROR: set DEST_FOLDERS_TT and DEST_FOLDERS_IG"); sys.exit(1)
    if len(DEST_FOLDERS_TT) != len(DEST_FOLDERS_IG):
        _log("ERROR: DEST_FOLDERS_TT and DEST_FOLDERS_IG lengths must match 1:1"); sys.exit(1)

    _print_config()
    d = _drive()

    # Resolve IDs
    try:
        input_real, _ = _resolve_shortcut(d, INPUT_DRIVE_FOLDER_ID)
    except HttpError as e:
        _log(f"ERROR resolving INPUT_DRIVE_FOLDER_ID: {e}"); sys.exit(1)

    tt_real: List[str] = []
    ig_real: List[str] = []
    for fid in DEST_FOLDERS_TT:
        try:
            real, _ = _resolve_shortcut(d, fid)
            tt_real.append(real)
        except HttpError as e:
            _log(f"[WARN] Skipping TT folder {fid}: {e}")
    for fid in DEST_FOLDERS_IG:
        try:
            real, _ = _resolve_shortcut(d, fid)
            ig_real.append(real)
        except HttpError as e:
            _log(f"[WARN] Skipping IG folder {fid}: {e}")

    if len(tt_real) != len(ig_real) or len(tt_real) == 0:
        _log("ERROR: Could not resolve equal count of TT and IG folders, or none resolved."); sys.exit(1)

    # 0) Wait for stable input
    _wait_for_stable_folder(
        d, input_real,
        window_sec=WAIT_FOR_STABILITY_SEC,
        timeout_sec=STABILITY_TIMEOUT_SEC,
        poll=POLL_INTERVAL_SEC
    )

    # 1) Wipe all TT & IG (Trash)
    if WIPE_DESTS:
        _log("[STEP] Wiping TT destinations (trash)…")
        for real in tt_real: _trash_all_in_folder(d, real)
        _log("[STEP] Wiping IG destinations (trash)…")
        for real in ig_real: _trash_all_in_folder(d, real)

    # 2) Load & filter source
    _log(f"[STEP] Listing input files from {input_real}…")
    files = _list_folder_files(d, input_real)
    _log(f"[INFO] Found {len(files)} items before filtering.")
    files = _only_videos(files)
    _log(f"[INFO] {len(files)} items after video/size filter.")
    if not files:
        _log("[OK] No eligible video files.")
        return

    files = _sort_files(files)
    if BATCH_PER_RUN > 0:
        files = files[:BATCH_PER_RUN]
        _log(f"[INFO] Limiting to first {BATCH_PER_RUN} files.")

    # 3) Balanced TT distribution heap: (count, tt_folder_id, idx)
    heap: List[Tuple[int, str, int]] = [(0 if DRY_RUN else _count_in_folder(d, fid), fid, idx)
                                        for idx, fid in enumerate(tt_real)]
    heapq.heapify(heap)
    _log(f"[INFO] Initial TT counts: {[(c, idx) for (c, _, idx) in heap]}")

    # 4) MOVE to TT, SHORTCUT to paired IG
    moved = 0
    shortcutted = 0

    for f in files:
        fname = f.get("name","(unnamed)")
        file_id = f.get("id","")
        parents = f.get("parents", []) or []

        cnt, tt_dst, idx = heapq.heappop(heap)
        ig_dst = ig_real[idx]

        try:
            remove_parent = input_real if input_real in parents else None
            _move_file(d, file_id, new_parent=tt_dst, remove_parent=remove_parent)
            moved += 1
            _log(f"[MOVE→TT#{idx+1}] {fname} → {tt_dst}")

            _shortcut_file(d, target_id=file_id, new_parent=ig_dst, name=fname)
            shortcutted += 1
            _log(f"[SHORTCUT→IG#{idx+1}] {fname} ⇢ {ig_dst}")

            cnt += 1
        except HttpError as e:
            _log(f"[ERROR] {fname}: {e}")
        finally:
            heapq.heappush(heap, (cnt, tt_dst, idx))

        time.sleep(0.03)

    _log(f"[DONE] moved_to_TT={moved} shortcuts_to_IG={shortcutted} | DRY_RUN={int(DRY_RUN)}")

if __name__ == "__main__":
    main()
