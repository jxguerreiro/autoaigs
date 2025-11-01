#!/usr/bin/env python3
import os, sys, time, json, io
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

# ======================= .env loader (same style) =======================
def load_env(dotenv_path: Path = Path(".env")):
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

load_env()

# ======================= ENV (after .env loaded) =======================
def _env_list(key: str) -> List[str]:
    raw = os.getenv(key, "") or ""
    return [x.strip() for x in raw.split(",") if x.strip()]

DRIVE_UPLOAD_FOLDER_ID = os.getenv("DRIVE_UPLOAD_FOLDER_ID", "")  # Final
DEST_FOLDERS_TT        = _env_list("DEST_FOLDERS_TT")
DEST_FOLDERS_IG        = _env_list("DEST_FOLDERS_IG")

GDRIVE_SA_JSON = os.getenv("GDRIVE_SA_JSON", "")
GDRIVE_SA_INFO = os.getenv("GDRIVE_SA_INFO", "")

DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# ======================= Google Drive setup =======================
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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

# ======================= Helpers =======================
def _resolve_shortcut(drive, fid: str) -> Tuple[str, Dict]:
    """Return (real_file_id, meta) for a file ID, following shortcuts."""
    meta = drive.files().get(
        fileId=fid,
        fields="id,name,mimeType,shortcutDetails,parents,driveId,trashed",
        supportsAllDrives=True
    ).execute()
    if meta.get("mimeType") == "application/vnd.google-apps.shortcut":
        tgt = meta.get("shortcutDetails", {}).get("targetId")
        if not tgt:
            raise RuntimeError(f"Shortcut {fid} has no targetId")
        meta = drive.files().get(
            fileId=tgt,
            fields="id,name,mimeType,parents,driveId,trashed",
            supportsAllDrives=True
        ).execute()
        return meta["id"], meta
    return meta["id"], meta

def _resolve_folder(drive, fid: str) -> str:
    """Resolve potential folder shortcuts to real folder ID and sanity check."""
    real, meta = _resolve_shortcut(drive, fid)
    if meta.get("mimeType") != "application/vnd.google-apps.folder":
        raise RuntimeError(f"{fid} is not a folder (mimeType={meta.get('mimeType')})")
    return real

def _list_folder(drive, folder_id: str, fields: str = "id,name,mimeType,parents,trashed") -> List[Dict]:
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

def _file_in_parent(drive, file_id: str, parent_id: str) -> bool:
    meta = drive.files().get(
        fileId=file_id, fields="id,parents", supportsAllDrives=True
    ).execute()
    return parent_id in (meta.get("parents") or [])

def _move_to_parent(drive, file_id: str, new_parent: str, remove_parent: Optional[str]) -> None:
    if DRY_RUN:
        _log(f"[DRY-RUN][MOVE] {file_id} : {remove_parent or ''} → {new_parent}")
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

def _collect_from_sources(drive, src_folders: List[str], final_folder: str, moved_set: Set[str], label: str):
    total_seen = total_moved = total_skipped = 0
    for src in src_folders:
        try:
            src_real = _resolve_folder(drive, src)
        except Exception as e:
            _log(f"[WARN] {label}: skip {src} (resolve failed): {e}")
            continue

        items = _list_folder(drive, src_real)
        _log(f"[{label}] {src_real}: {len(items)} item(s) to check")
        for it in items:
            fid = it["id"]
            name = it.get("name", "(unnamed)")
            mime = it.get("mimeType", "")
            parents = it.get("parents", []) or []
            total_seen += 1

            # If it's a shortcut here, move the TARGET instead of the shortcut file
            if mime == "application/vnd.google-apps.shortcut":
                try:
                    tgt_id, tgt = _resolve_shortcut(drive, fid)
                except Exception as e:
                    _log(f"[{label}][SKIP] shortcut {fid}/{name}: cannot resolve target: {e}")
                    total_skipped += 1
                    continue
                if tgt.get("trashed"):
                    _log(f"[{label}][SKIP] {name}: target is trashed")
                    total_skipped += 1
                    continue
                if tgt_id in moved_set:
                    _log(f"[{label}][SKIP] {name}: target already moved")
                    total_skipped += 1
                    continue
                if _file_in_parent(drive, tgt_id, final_folder):
                    _log(f"[{label}][SKIP] {name}: target already in Final")
                    moved_set.add(tgt_id)
                    total_skipped += 1
                    continue

                remove_parent = None
                # Try to remove the source folder parent if present (either on shortcut or on target)
                for p in (parents + (tgt.get("parents") or [])):
                    if p == src_real:
                        remove_parent = p
                        break

                try:
                    _move_to_parent(drive, tgt_id, new_parent=final_folder, remove_parent=remove_parent)
                    total_moved += 1
                    moved_set.add(tgt_id)
                    _log(f"[{label}][MOVE TARGET] {name} ({tgt_id}) → Final")
                except HttpError as e:
                    _log(f"[{label}][ERROR] moving target for shortcut '{name}': {e}")
                continue

            # Normal file: move it
            if it.get("trashed"):
                _log(f"[{label}][SKIP] {name}: trashed")
                total_skipped += 1
                continue
            if fid in moved_set:
                _log(f"[{label}][SKIP] {name}: already moved")
                total_skipped += 1
                continue
            if _file_in_parent(drive, fid, final_folder):
                _log(f"[{label}][SKIP] {name}: already in Final")
                moved_set.add(fid)
                total_skipped += 1
                continue

            remove_parent = src_real if src_real in parents else None
            try:
                _move_to_parent(drive, fid, new_parent=final_folder, remove_parent=remove_parent)
                total_moved += 1
                moved_set.add(fid)
                _log(f"[{label}][MOVE] {name} ({fid}) → Final")
            except HttpError as e:
                _log(f"[{label}][ERROR] moving '{name}': {e}")

        time.sleep(0.02)
    _log(f"[{label}] done: seen={total_seen} moved={total_moved} skipped={total_skipped}")

# ======================= Main =======================
def main():
    if not DRIVE_UPLOAD_FOLDER_ID:
        _log("ERROR: set DRIVE_UPLOAD_FOLDER_ID (Final folder id)"); sys.exit(1)
    if not DEST_FOLDERS_TT and not DEST_FOLDERS_IG:
        _log("ERROR: set DEST_FOLDERS_TT and/or DEST_FOLDERS_IG"); sys.exit(1)

    _log(f"[CONFIG] Final={DRIVE_UPLOAD_FOLDER_ID} | TT={len(DEST_FOLDERS_TT)} IG={len(DEST_FOLDERS_IG)} DRY_RUN={int(DRY_RUN)}")

    d = _drive()

    # Resolve Final
    try:
        final_real = _resolve_folder(d, DRIVE_UPLOAD_FOLDER_ID)
    except Exception as e:
        _log(f"ERROR resolving Final folder: {e}")
        sys.exit(1)

    moved_set: Set[str] = set()

    # TT first (moves the real files), then IG (shortcuts will point to those files; non-shortcuts in IG also move)
    if DEST_FOLDERS_TT:
        _collect_from_sources(d, DEST_FOLDERS_TT, final_real, moved_set, label="TT")
    if DEST_FOLDERS_IG:
        _collect_from_sources(d, DEST_FOLDERS_IG, final_real, moved_set, label="IG")

    _log("[DONE] Consolidation complete.")

if __name__ == "__main__":
    main()
