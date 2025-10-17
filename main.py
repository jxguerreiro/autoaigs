#!/usr/bin/env python3
import os, re, json, time, random, shlex, subprocess, tempfile, hashlib, shutil
from pathlib import Path

# ================== CONFIG (FOLDERS) ==================
GREENSCREEN_DIR = Path("greenscreen")                         # input videos (directory)
SCRIPTS_DIR     = Path("/home/joao/farm_scripts/scripts")     # NUMBER.txt scripts (directory)
OUT_DIR         = Path("output_split")                        # working scratch
FINAL_VIDS_DIR  = Path("final videos")                        # <— final outputs land here

# B-roll libraries
HOOKS_DIR  = Path("/home/joao/farm_scripts/broll_library/hooks")
BODIES_DIR = Path("/home/joao/farm_scripts/broll_library/bodies")

# explicit override folders (matched from the end backwards)
AMAZON_DIR = Path("/home/joao/farm_scripts/broll_library/amazon")
BUGMD_DIR  = Path("/home/joao/farm_scripts/broll_library/bugMD")
SPRAY_DIR  = Path("/home/joao/farm_scripts/broll_library/spray")
RASHES_DIR = Path("/home/joao/farm_scripts/broll_library/bodies/rashes skin collars")

# whisper
WHISPER_PY     = "/home/joao/miniconda3/envs/whisper_env/bin/python"
WHISPER_MODEL  = os.getenv("WHISPER_MODEL","medium")
USE_FASTER_WHISPER = os.getenv("USE_FASTER_WHISPER","1") not in ("0","false","False")
FW_DEVICE       = os.getenv("FASTER_WHISPER_DEVICE", os.getenv("WHISPER_DEVICE","cuda"))
FW_COMPUTE_TYPE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE","int8_float16" if FW_DEVICE=="cuda" else "int8")
FP16            = False

# ffmpeg
USE_HWACCEL_CUDA = os.getenv("USE_HWACCEL_CUDA","1") not in ("0","false","False")

# silence clamp
MIN_SILENCE_SEC   = float(os.getenv("MIN_SILENCE_SEC","0.7"))
KEEP_SILENCE_SEC  = 0.15
SILENCE_NOISE_DB  = float(os.getenv("SILENCE_NOISE_DB","-35.0"))

# final speed
FINAL_SPEED       = float(os.getenv("FINAL_SPEED","1.1"))
# ======================================================

# ---------- logging / exec ----------
_t0 = time.time()
def log(msg): print(f"[{time.time()-_t0:7.2f}s] {msg}")

def run(cmd, echo=True):
    if echo:
        print("\n── ffcmd ──")
        print(cmd)
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "Command failed")
    return p

def ff_has_nvenc():
    try:
        p = subprocess.run(["ffmpeg","-hide_banner","-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return "h264_nvenc" in p.stdout
    except Exception:
        return False

NVENC_OK = ff_has_nvenc()
def vcodec_flags():
    return "-c:v h264_nvenc -preset p4 -rc vbr -cq 19 -b:v 0 -pix_fmt yuv420p" if NVENC_OK else "-c:v libx264 -pix_fmt yuv420p"
def hwaccel_prefix(): return "-hwaccel cuda " if (NVENC_OK and USE_HWACCEL_CUDA) else ""

# ---------- probes ----------
_ff_cache = {}
def ffprobe_stream(path):
    if path in _ff_cache: return _ff_cache[path]
    j = run(f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of json "{path}"', echo=False).stdout
    info = json.loads(j)["streams"][0]
    w, h = info["width"], info["height"]
    fr = info.get("r_frame_rate","30/1").split("/")
    fps = float(fr[0])/(float(fr[1]) or 1.0)
    dur = float(info.get("duration", 0.0) or 0.0)
    _ff_cache[path] = (w,h,fps,dur)
    return _ff_cache[path]

def has_audio_stream(path: str) -> bool:
    try:
        j = run(f'ffprobe -v error -select_streams a:0 -show_entries stream=index -of json "{path}"', echo=False).stdout
        return bool(json.loads(j).get("streams"))
    except Exception:
        return False

# ---------- text utils ----------
import re as _re, unicodedata as _ud
def strip_accents(s: str) -> str:
    return "".join(c for c in _ud.normalize("NFD", s) if _ud.category(c) != "Mn")
def light_stem(w: str) -> str:
    w = w.lower()
    if len(w) <= 3: return w
    if len(w) > 4 and w.endswith("ies"): return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"): return w[:-3]
    if len(w) > 4 and w.endswith("ed"):  return w[:-2]
    if len(w) > 3 and w.endswith("es"):  return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"): return w[:-1]
    return w
def tokenize_norm(text: str):
    return [light_stem(t) for t in _re.findall(r"[a-z0-9]+", strip_accents(text).lower())]
def split_sentences_period(text):
    return [s.strip() for s in _re.split(r'\.\s*', text) if s.strip()]

# ---------- align script to words ----------
def pick_tail_tokens(sent, max_tail=6):
    toks = tokenize_norm(sent)
    return toks[-max_tail:] if toks else []

def find_sentence_end_time(words, start_idx, tail_tokens, fallback_last_word, max_look=5000):
    if not words: return 0.5, 0
    n = len(words); end_idx = min(n-1, start_idx + max_look); tail_len = len(tail_tokens)
    def last_tok(w):
        t = tokenize_norm(w.get("word",""))
        return t[-1] if t else ""
    def match_k_at(i, k):
        if i - k + 1 < start_idx: return False
        for j in range(k):
            if last_tok(words[i - k + 1 + j]) != tail_tokens[tail_len - k + j]:
                return False
        return True
    for k in [6,5,4,3,2,1]:
        if tail_len < k: continue
        for i in range(start_idx + k - 1, end_idx + 1):
            if match_k_at(i, k): return words[i]["end"], i + 1
    if fallback_last_word:
        for i in range(start_idx, end_idx + 1):
            if last_tok(words[i]) == fallback_last_word: return words[i]["end"], i + 1
    return words[end_idx]["end"], end_idx + 1

def build_segments_from_script(words, script_sentences):
    segments=[]; cur_idx=0; cur_time=0.0
    for sent in script_sentences:
        tail = pick_tail_tokens(sent, max_tail=6)
        fallback = tail[-1] if tail else None
        end_time, next_idx = find_sentence_end_time(words, cur_idx, tail, fallback)
        start_t = max(cur_time, 0.0)
        end_t   = max(end_time, start_t + 0.40)
        segments.append({"text":sent, "start":start_t, "end":end_t})
        cur_idx = next_idx; cur_time = end_t
    if segments: segments[-1]["end"] += 0.10
    return segments

# ---------- whisper ----------
def extract_audio_to_wav(input_video, out_wav):
    run(f'ffmpeg -y -nostdin -loglevel error {hwaccel_prefix()}-i "{input_video}" -vn -ac 1 -ar 16000 -c:a pcm_s16le "{out_wav}"')

def _extract_json_array(mixed_text: str):
    m = _re.search(r'\[.*\]', mixed_text.strip(), flags=_re.DOTALL)
    if not m: raise ValueError("No JSON array from whisper.")
    return json.loads(m.group(0))

def whisper_words_fallback(audio_wav_path):
    code = r'''
import json, sys
try:
    import whisper
except Exception as e:
    print("Whisper not available:", e, file=sys.stderr); sys.exit(2)
audio = sys.argv[1]; model_name = sys.argv[2]; fp16 = (sys.argv[3].lower()=="true")
model = whisper.load_model(model_name)
res = model.transcribe(audio, task="transcribe", word_timestamps=True, verbose=False, fp16=fp16, condition_on_previous_text=False, temperature=0.0)
out=[]
for s in res.get("segments", []):
    for w in s.get("words", []) or []:
        txt=(w.get("word") or "").strip(); st=w.get("start"); et=w.get("end")
        if txt and st is not None and et is not None: out.append({"word":txt,"start":float(st),"end":float(et)})
print(json.dumps(out))
'''
    p = subprocess.run([WHISPER_PY, "-c", code, str(audio_wav_path), WHISPER_MODEL, "true" if FP16 else "false"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0: raise RuntimeError(p.stderr)
    return _extract_json_array(p.stdout)

def faster_whisper_words(audio_wav_path):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        log(f"faster-whisper import failed: {e} — using whisper CPU")
        return whisper_words_fallback(audio_wav_path)
    model = WhisperModel(WHISPER_MODEL, device=FW_DEVICE, compute_type=FW_COMPUTE_TYPE)
    words=[]
    segments, _ = model.transcribe(audio_wav_path, task="transcribe", vad_filter=True, beam_size=1,
                                   word_timestamps=True, temperature=0.0, condition_on_previous_text=False)
    for s in segments:
        for w in (s.words or []):
            txt=(w.word or "").strip()
            if txt: words.append({"word":txt, "start":float(w.start), "end":float(w.end)})
    return words

def whisper_words(audio_wav_path):
    return faster_whisper_words(audio_wav_path) if USE_FASTER_WHISPER else whisper_words_fallback(audio_wav_path)

# ---------- subtitles ----------
def write_centered_ass_chunks(chunks, seg_dur, out_ass_path, font="Arial", size=64, outline=4, shadow=0):
    def ts(t):
        t = max(0.0, float(t)); h=int(t//3600); m=int((t%3600)//60); s=int(t%60); cs=int(round((t-int(t))*100))
        return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: TikTok,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},{shadow},5,20,20,0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for c in chunks:
            safe = c["text"].replace("{","(").replace("}",")")
            f.write(f"Dialogue: 0,{ts(c['start'])},{ts(c['end'])},TikTok,,0,0,0,,{safe}\n")

def split_chunks_by_limits_words(words_list, max_chars=26, max_words=6, min_chunk_dur=0.25):
    chunks=[]; cur_tokens=[]; cur_start=None; last_end=None
    def flush():
        nonlocal cur_tokens, cur_start, last_end
        if not cur_tokens: return
        text=" ".join(t["txt"] for t in cur_tokens).strip()
        st=cur_start; en = last_end if last_end is not None else st + min_chunk_dur
        en=max(st+min_chunk_dur, en)
        chunks.append({"text":text,"start":st,"end":en})
        cur_tokens=[]; cur_start=None
    for w in words_list:
        token=w["txt"]; candidate=(" ".join([*(t["txt"] for t in cur_tokens), token]) if cur_tokens else token)
        if cur_tokens and (len(cur_tokens)+1>max_words or len(candidate)>max_chars): flush()
        if not cur_tokens: cur_start=w["start"]
        cur_tokens.append({"txt":token}); last_end=w["end"]
    flush(); return chunks

def build_audio_chunks_for_window(words, win_start, win_end, max_chars=26, max_words=6, min_chunk_dur=0.25):
    sel=[]
    for w in words:
        if w["start"] >= win_end: break
        if w["end"] <= win_start: continue
        sel.append({"txt":w["word"].strip(), "start":max(win_start, w["start"]), "end":min(win_end, w["end"])})
    if not sel:
        return [{"text":"","start":0.0,"end":max(min_chunk_dur, win_end-win_start)}]
    chunks_abs = split_chunks_by_limits_words(sel, max_chars=max_chars, max_words=max_words, min_chunk_dur=min_chunk_dur)
    if chunks_abs: chunks_abs[-1]["end"] = max(chunks_abs[-1]["end"], win_end)
    return [{"text":c["text"], "start":c["start"]-win_start, "end":c["end"]-win_start} for c in chunks_abs]

# ---------- media helpers ----------
MEDIA_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
MEDIA_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
def is_image(p: Path) -> bool: return p.suffix.lower() in MEDIA_IMAGE_EXTS
def is_video(p: Path) -> bool: return p.suffix.lower() in MEDIA_VIDEO_EXTS
ALL_MEDIA_EXTS = MEDIA_VIDEO_EXTS | MEDIA_IMAGE_EXTS

def list_media_recursive(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in (MEDIA_VIDEO_EXTS | MEDIA_IMAGE_EXTS)])

def get_clip_dur(p: Path, fallback_for_image: float = 36000.0):
    if is_image(p): return float(fallback_for_image)
    try:
        _,_,_,dur = ffprobe_stream(str(p)); return max(0.1, float(dur))
    except Exception:
        return 0.1

# ---------- bodies folder selection ----------
def folder_tokens(name: str): return set(tokenize_norm(name))

def best_bodies_folder_by_keywords(sentence: str):
    """Pick a subfolder under BODIES_DIR by token overlap (fallback: most populated)."""
    sent_tokens = set(tokenize_norm(sentence))
    subfolders = [d for d in BODIES_DIR.iterdir() if d.is_dir()]
    best, best_score = None, -1.0
    for d in subfolders:
        ft = folder_tokens(d.name)
        inter = len(sent_tokens & ft)
        if inter > best_score:
            best, best_score = d, inter
    if best is None and subfolders:
        best = max(subfolders, key=lambda d: len(list_media_recursive(d)))
    return best

def plan_broll_sequence_in_folder(folder: Path, need_sec: float, avoid_first: Path|None = None,
                                  min_second_clip: float = 3.0, min_first_clip: float = 1.0):
    files = list_media_recursive(folder) if folder and folder.is_dir() else []
    if not files:
        return []
    rnd = random.Random()
    choices = files[:]
    if avoid_first and avoid_first in choices and len(choices) > 1:
        choices = [c for c in choices if c != avoid_first]
    p1 = rnd.choice(choices); d1 = get_clip_dur(p1)
    if d1 >= need_sec: return [(p1, need_sec)]
    p2 = rnd.choice(files); d2 = get_clip_dur(p2)
    remaining = need_sec - d1
    if remaining < min_second_clip:
        clip2 = min(d2, max(min_second_clip, remaining))
        clip1 = max(min_first_clip, min(d1, need_sec - clip2))
        planned=[(p1, clip1), (p2, clip2)]
    else:
        planned=[(p1, d1), (p2, min(d2, remaining))]
    total = sum(t for _,t in planned); guard=0
    while total + 0.01 < need_sec and guard < 32:
        p = rnd.choice(files); d = get_clip_dur(p); take = min(d, need_sec-total)
        planned.append((p, take)); total += take; guard += 1
    if total > need_sec and planned:
        over = total - need_sec
        last_p, t_last = planned[-1]
        planned[-1] = (last_p, max(0.1, t_last - over))
    return planned

# ---------- renderers ----------
def render_segment_single_call(out_path, base_seq, person_mov, seg_start, seg_end, place_right, out_size, fps, ass_path=None):
    W,H = out_size
    cmd = ["ffmpeg","-y","-nostdin","-loglevel","error"]
    idx = 0; base_labels=[]

    if not base_seq:
        need_t = max(0.25, seg_end-seg_start)
        cmd += ["-f","lavfi","-t",f"{need_t:.6f}","-i",f"color=c=black:s={W}x{H}:r={fps:.2f}"]
        base_labels.append((idx, need_t, True, True)); idx += 1
    else:
        for src, play_t in base_seq:
            play_t = max(0.1, float(play_t))
            if is_image(src):
                cmd += ["-loop","1","-t",f"{play_t:.6f}","-i",str(src)]
                base_labels.append((idx, play_t, True, False))
            else:
                cmd += ["-i", str(src)]
                base_labels.append((idx, play_t, False, False))
            idx += 1

    cmd += ["-i", str(person_mov)]
    person_idx = idx

    chains=[]; outs=[]
    for (i, play_t, is_img, is_black) in base_labels:
        ops=[]
        if not is_black:
            ops += [f"fps={fps:.2f}", f"scale={W}:{H}:force_original_aspect_ratio=decrease", f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2"]
        ops += [f"trim=end={play_t:.6f}", "setpts=PTS-STARTPTS"]
        chains.append(f"[{i}:v]{','.join(ops)}[bb{i}]"); outs.append(f"[bb{i}]")

    base_out = outs[0] if len(outs)==1 else "".join(outs)+f"concat=n={len(outs)}:v=1:a=0[base]"
    if len(outs)>1: chains.append(base_out); base_out="[base]"

    start_t=max(0.0, seg_start); end_t=max(start_t+0.25, seg_end)
    chains.append(f"[{person_idx}:v]trim=start={start_t:.6f}:end={end_t:.6f},setpts=PTS-STARTPTS,format=rgba,scale=iw*1.0:-1:flags=bicubic,format=rgba[fg]")

    x_expr = f"W-w-0" if place_right else "0"
    y_expr = f"H-h-0"
    chains.append(f"{base_out}[fg]overlay=x={x_expr}:y={y_expr}:format=auto:eval=frame[vv]")

    map_video="[vv]"
    if ass_path:
        ass = Path(ass_path).as_posix().replace("\\","/")
        chains.append(f"[vv]subtitles='{ass}'[vout]")
        map_video="[vout]"

    cmd += ["-filter_complex",";".join(chains),"-map",map_video,"-an"]
    cmd += vcodec_flags().split(); cmd += [str(out_path)]
    run(" ".join(shlex.quote(c) for c in cmd), echo=True)

def render_outro_with_subs(out_path, input_video, ss, to, W, H, fps, ass_path):
    ass = Path(ass_path).as_posix().replace("\\","/")
    filt = (
        f"[0:v]trim=start={ss:.6f}:end={to:.6f},setpts=PTS-STARTPTS,"
        f"scale={W}:{H},fps={fps:.02f},subtitles='{ass}'[v]"
    )
    run(
        f'ffmpeg -y -nostdin -loglevel error {hwaccel_prefix()}'
        f'-i "{input_video}" -an -filter_complex "{filt}" -map "[v]" {vcodec_flags()} "{out_path}"'
    )

# ---------- silence clamp ----------
def _parse_silencedetect(log_text):
    sil=[]; cur={}
    for line in log_text.splitlines():
        line=line.strip()
        if "silence_start:" in line:
            try: cur={"start":float(line.split("silence_start:")[1].strip())}
            except: pass
        elif "silence_end:" in line and "silence_duration:" in line and "start" in cur:
            try:
                parts=line.split("silence_end:")[1].split("|")
                end=float(parts[0].strip()); dur=float(parts[1].split("silence_duration:")[1].strip())
                sil.append((cur["start"], end, dur)); cur={}
            except: pass
    return sil

def shrink_silences_keep(in_path: str, out_path: str,
                         min_silence=0.7, keep_silence=0.15, noise_db=-35.0):
    p = subprocess.run(shlex.split(
        f'ffmpeg -hide_banner -nostdin -i "{in_path}" -af silencedetect=noise={noise_db}dB:d={min_silence} -f null -'
    ), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    silences = _parse_silencedetect(p.stderr)
    j = run(f'ffprobe -v error -show_entries format=duration -of json "{in_path}"', echo=False).stdout
    T = float(json.loads(j)["format"]["duration"])
    has_aud = has_audio_stream(in_path)
    EPS=1e-3; windows=[]; t=0.0
    for s,e,d in silences:
        s=max(0.0,s); e=max(s,e)
        if s-t>EPS: windows.append((t,s))
        if d>=min_silence:
            head=min(e, s+keep_silence)
            if head-s>EPS: windows.append((s,head))
            t=e
        else:
            if e-s>EPS: windows.append((s,e))
            t=e
    if T-t>EPS: windows.append((t,T))
    if len(windows)<=1:
        run(f'ffmpeg -y -nostdin -loglevel error -i "{in_path}" -c copy -movflags +faststart "{out_path}"', echo=False); return
    parts_v=[]; parts_a=[]
    for i,(a,b) in enumerate(windows):
        a=max(0.0,a); b=max(a+0.0005,b)
        parts_v.append(f'[0:v]trim=start={a:.6f}:end={b:.6f},setpts=PTS-STARTPTS[v{i}]')
        if has_aud: parts_a.append(f'[0:a]atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS[a{i}]')
    v_inputs=''.join(f'[v{i}]' for i in range(len(windows)))
    if has_aud:
        a_inputs=''.join(f'[a{i}]' for i in range(len(windows)))
        filt=';'.join(parts_v+parts_a+[f'{v_inputs}concat=n={len(windows)}:v=1:a=0[v]', f'{a_inputs}concat=n={len(windows)}:v=0:a=1[a]'])
        run(f'ffmpeg -y -nostdin -loglevel error -i "{in_path}" -filter_complex "{filt}" -map "[v]" -map "[a]" {vcodec_flags()} -c:a aac -b:a 192k -movflags +faststart "{out_path}"', echo=False)
    else:
        filt=';'.join(parts_v+[f'{v_inputs}concat=n={len(windows)}:v=1:a=0[v]'])
        run(f'ffmpeg -y -nostdin -loglevel error -i "{in_path}" -filter_complex "{filt}" -map "[v]" -an {vcodec_flags()} -movflags +faststart "{out_path}"', echo=False)

# ---------- speed up ----------
def speed_up_final(in_path: str, out_path: str, speed=1.1):
    if has_audio_stream(in_path):
        run(f'ffmpeg -y -nostdin -loglevel error -i "{in_path}" -filter_complex "[0:v]setpts=PTS/{speed}[v];[0:a]atempo={speed}[a]" -map "[v]" -map "[a]" {vcodec_flags()} -c:a aac -b:a 192k -movflags +faststart "{out_path}"')
    else:
        run(f'ffmpeg -y -nostdin -loglevel error -i "{in_path}" -vf "setpts=PTS/{speed}" {vcodec_flags()} -an -movflags +faststart "{out_path}"')

# ---------- transparent person maker ----------
def fast_transparent_segment(input_path: str, output_mov: str, target_height: int = 720, threshold: float = 0.52,
                             blur_ksize: int = 25, alpha_soften: float = 0.10, ema_decay: float = 0.80,
                             ffmpeg_timeout: int = 1200):
    import cv2, numpy as np, mediapipe as mp, subprocess, shlex
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened(): raise FileNotFoundError(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(3)); src_h = int(cap.get(4))
    if target_height and target_height < src_h:
        scale = target_height / src_h; W,H = int(round(src_w*scale)), int(round(src_h*scale))
    else:
        W,H = src_w, src_h
    ffmpeg_cmd = f'ffmpeg -y -nostdin -loglevel error -f rawvideo -pix_fmt rgba -s {W}x{H} -r {fps:.02f} -i - -an -c:v prores_ks -profile:v 4 -pix_fmt yuva444p10le -movflags +faststart "{output_mov}"'
    proc = subprocess.Popen(shlex.split(ffmpeg_cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)); last_a=None
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok: break
            if (W,H)!=(src_w,src_h): frame_bgr = cv2.resize(frame_bgr,(W,H),interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mask = mp_selfie.process(frame_rgb).segmentation_mask.astype("float32")
            if blur_ksize>0:
                k=blur_ksize|1; mask=cv2.GaussianBlur(mask,(k,k),0)
            mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            a_cur = np.clip((mask - threshold)/max(1e-6,(1.0-threshold)), 0.0, 1.0)
            a_ema = a_cur if last_a is None else (ema_decay*last_a + (1.0-ema_decay)*a_cur)
            last_a=a_ema
            if alpha_soften>0: a_ema=cv2.GaussianBlur(a_ema,(0,0),1.0)
            alpha_u=(np.clip(a_ema,0,1)*255.0).astype("uint8")
            rgb_u8=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype("uint8")
            proc.stdin.write(np.dstack((rgb_u8, alpha_u)).tobytes())
    finally:
        cap.release()
        if proc.stdin:
            try: proc.stdin.close()
            except: pass
            proc.stdin=None
        try:
            proc.communicate(timeout=ffmpeg_timeout)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.communicate(); raise RuntimeError("ffmpeg timed out.")
        if proc.returncode!=0: raise RuntimeError("ffmpeg failed writing MOV.")

# ---------- end→back folder override ----------
def choose_broll_folder_for_index(i: int, N: int, sentence_text: str, last_first_clip_path, hooks_mode: bool):
    """
    End→back blocks (highest priority, timings unchanged):
      - last 2 (from_end: 0,1) → AMAZON_DIR
      - prev 2 (from_end: 2,3) → BUGMD_DIR
      - prev 2 (from_end: 4,5) → SPRAY_DIR
    Then:
      - if i == 7 (8th sentence overall), use RASHES_DIR (unless overridden above).
    Otherwise fallback to semantic folder under BODIES_DIR.
    """
    from_end = (N - 1) - i
    if from_end in (0, 1) and AMAZON_DIR.exists(): return AMAZON_DIR
    if from_end in (2, 3) and BUGMD_DIR.exists():   return BUGMD_DIR
    if from_end in (4, 5) and SPRAY_DIR.exists():   return SPRAY_DIR
    if i == 7 and RASHES_DIR.exists():              return RASHES_DIR
    if hooks_mode: return HOOKS_DIR if HOOKS_DIR.exists() else BODIES_DIR
    folder = best_bodies_folder_by_keywords(sentence_text)
    return folder if (folder and folder.exists()) else BODIES_DIR

# ---------- naming helpers (collision-safe tagging) ----------
VALID_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

def _extract_number_anywhere(stem: str) -> str | None:
    m = re.search(r'(\d+)', stem)
    return m.group(1) if m else None

def _extract_number_from_scriptish(stem: str) -> str | None:
    m = re.search(r'(?i)script[_\- ]*(\d+)', stem)
    return m.group(1) if m else None

def _slugify(s: str, max_len: int = 60) -> str:
    s = strip_accents(s)
    s = re.sub(r'[^a-zA-Z0-9]+', '-', s).strip('-').lower()
    if not s: s = "clip"
    return s[:max_len]

def _short_stable_id(p: Path, length: int = 4) -> str:
    h = hashlib.sha1(str(p.resolve()).encode('utf-8')).hexdigest()
    return h[:length]

def _already_tagged(fname: str) -> bool:
    return re.match(r'^\d+__.+\.[A-Za-z0-9]+$', fname) is not None

def normalize_greenscreen_filenames(gs_dir: Path, scripts_dir: Path) -> dict:
    """
    Tag each video with its script number: '<num>__<slugified-original-stem>.<ext>'
    - If file already starts with '<num>__', it's left as-is.
    - If multiple videos map to the same target, append short '-<hash4>' (and -2, -3 if needed).
    Only rename if scripts/<num>.txt exists.
    """
    renames = {}
    if not gs_dir.exists():
        return renames

    for p in gs_dir.iterdir():
        if not p.is_file(): continue
        ext = p.suffix.lower()
        if ext not in VALID_VIDEO_EXTS: continue

        if _already_tagged(p.name):
            continue

        num = _extract_number_from_scriptish(p.stem) or _extract_number_anywhere(p.stem)
        if not num:
            continue
        if not (scripts_dir / f"{num}.txt").exists():
            continue

        slug = _slugify(p.stem)
        base = f"{num}__{slug}{ext}"
        target = gs_dir / base

        if target.exists():
            suf = _short_stable_id(p)
            target = gs_dir / f"{num}__{slug}-{suf}{ext}"
            counter = 2
            while target.exists() and counter < 1000:
                target = gs_dir / f"{num}__{slug}-{suf}-{counter}{ext}"
                counter += 1

        try:
            p.rename(target)
            renames[str(p)] = str(target)
            log(f"🔤 tagged: {p.name} → {target.name}")
        except Exception as e:
            log(f"❌ rename failed for {p.name}: {e}")

    return renames

# ---------- discovery ----------
def _extract_number_from_tagged_or_any(stem: str) -> str | None:
    m = re.match(r'^(\d+)__', stem)
    if m: return m.group(1)
    return _extract_number_anywhere(stem)

def find_processable_videos(gs_dir: Path, scripts_dir: Path):
    """
    Return list of (video_path, script_path) where:
      - video is a file with a detectable number (tagged or anywhere in name)
      - matching scripts/<num>.txt exists
    """
    vids = []
    if not gs_dir.exists():
        return vids
    for p in sorted(gs_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in VALID_VIDEO_EXTS):
            continue
        num = _extract_number_from_tagged_or_any(p.stem)
        if not num: 
            continue
        s = scripts_dir / f"{num}.txt"
        if s.exists():
            vids.append((p, s))
    return vids

# ---------- final mover & cleaner ----------
def _unique_dest(base_path: Path) -> Path:
    """Return a non-colliding destination path (adds -1, -2, ... if needed)."""
    if not base_path.exists():
        return base_path
    stem, ext = base_path.stem, base_path.suffix
    i = 1
    while True:
        cand = base_path.with_name(f"{stem}-{i}{ext}")
        if not cand.exists():
            return cand
        i += 1

def move_to_final(final_out: Path, final_dir: Path) -> Path:
    final_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(final_dir / final_out.name)
    shutil.move(str(final_out), str(dest))
    return dest

def clean_output_dir(out_dir: Path):
    """Hard-clean the output scratch directory (files + subfolders)."""
    if not out_dir.exists():
        return
    for item in out_dir.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception:
            pass

# ---------- per-video processing ----------
def process_one(input_video: Path, script_txt: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stem = input_video.stem
    W,H,fps,_ = ffprobe_stream(str(input_video))

    # extract audio → words
    raw_audio = OUT_DIR / f"{stem}_raw.wav"
    log(f"🎧 extracting audio… ({input_video.name})")
    extract_audio_to_wav(str(input_video), raw_audio)
    log("🗣️ transcribing…")
    words = whisper_words(str(raw_audio))

    # script → windows
    script_text = Path(script_txt).read_text(encoding="utf-8")
    script_sent = split_sentences_period(script_text)
    assert script_sent, f"No sentences in script {script_txt}."
    segments = build_segments_from_script(words, script_sent)
    log(f"built {len(segments)} segments for {stem}")

    # transparent person (cache per video)
    trans_mov = OUT_DIR / f"{stem}_no_bg.mov"
    if not trans_mov.exists():
        log("🎨 making transparent person MOV… (first run only)")
        fast_transparent_segment(str(input_video), str(trans_mov), target_height=720, threshold=0.52)

    # render segments
    seg_files=[]; alt_right=True; last_first_clip=None
    N = len(segments)
    for i,p in enumerate(segments):
        seg_out = OUT_DIR / f"{stem}_seg_{i:03d}.mp4"
        dur = max(0.25, p["end"]-p["start"])

        with tempfile.TemporaryDirectory() as td_sub:
            ass_path = Path(td_sub)/"seg.ass"
            rel_chunks = build_audio_chunks_for_window(words, p["start"], p["end"])
            write_centered_ass_chunks(rel_chunks, dur, ass_path)

            if i == N-1:
                ss = max(0.0, p["start"]); to = max(ss + 0.05, p["end"])
                log(f"🎬 {stem} seg {i:03d} OUTRO fullscreen (ss={ss:.3f}, to={to:.3f})")
                render_outro_with_subs(str(seg_out), str(input_video), ss, to, W, H, fps, ass_path)
            else:
                hooks_mode = (i < 2)
                folder = choose_broll_folder_for_index(i, N, p["text"], last_first_clip, hooks_mode)
                base_seq = plan_broll_sequence_in_folder(folder, dur, avoid_first=last_first_clip) if folder else []
                if base_seq:
                    last_first_clip = base_seq[0][0]
                tot = sum(t for _,t in base_seq) if base_seq else 0.0
                if base_seq and abs(tot - dur) > 1e-3:
                    p_last, t_last = base_seq[-1]
                    base_seq[-1] = (p_last, max(0.1, t_last + (dur - tot)))
                render_segment_single_call(
                    out_path=str(seg_out),
                    base_seq=base_seq,
                    person_mov=str(trans_mov),
                    seg_start=p["start"], seg_end=p["end"],
                    place_right=alt_right,
                    out_size=(W,H), fps=fps,
                    ass_path=ass_path
                )
                label = "HOOK_BODY" if i < 2 else "BODY"
                chosen_folder_name = folder.name if folder else "None"
                log(f"🎬 {stem} seg {i:03d} {label} folder={chosen_folder_name} side={'RIGHT' if alt_right else 'LEFT'} {p['start']:.2f}→{p['end']:.2f}")
                alt_right = not alt_right

        seg_files.append(seg_out)

    # concat (stream copy)
    concat_txt = OUT_DIR / f"{stem}_concat_list.txt"
    with open(concat_txt,"w") as f:
        for s in seg_files: f.write(f"file '{s.resolve().as_posix()}'\n")
    concat = OUT_DIR / f"{stem}_concat.mp4"
    run(f'ffmpeg -y -nostdin -loglevel error -f concat -safe 0 -i "{concat_txt}" -c copy -movflags +faststart "{concat}"')

    # narration (from original) + mux
    narration = OUT_DIR / f"{stem}_narration.m4a"
    run(f'ffmpeg -y -nostdin -loglevel error -i "{input_video}" -vn -c:a aac -b:a 192k "{narration}"')
    muxed = OUT_DIR / f"{stem}_muxed.mp4"
    run(f'ffmpeg -y -nostdin -loglevel error -i "{concat}" -i "{narration}" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart "{muxed}"')

    # clamp silences
    clamped = OUT_DIR / f"{stem}_clamped.mp4"
    log("🔇 clamping long silences to 0.15s…")
    shrink_silences_keep(str(muxed), str(clamped), min_silence=MIN_SILENCE_SEC, keep_silence=KEEP_SILENCE_SEC, noise_db=SILENCE_NOISE_DB)

    # final speed
    final_out = OUT_DIR / f"{stem}_FINAL.mp4"
    log(f"⏩ applying global {FINAL_SPEED}x speed…")
    speed_up_final(str(clamped), str(final_out), speed=FINAL_SPEED)

    # move final → "final videos" and clean OUT_DIR
    dest = move_to_final(final_out, FINAL_VIDS_DIR)
    log(f"📦 moved final → {dest}")
    clean_output_dir(OUT_DIR)
    log("🧹 cleaned output_split/")

    return dest

# ---------- batch runner (always) ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_VIDS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Normalize filenames in greenscreen (tag with NUMBER when script exists)
    normalize_greenscreen_filenames(GREENSCREEN_DIR, SCRIPTS_DIR)

    # 2) Discover all processable (video, matching script)
    pairs = find_processable_videos(GREENSCREEN_DIR, SCRIPTS_DIR)
    if not pairs:
        gs_list = [p.name for p in GREENSCREEN_DIR.iterdir()] if GREENSCREEN_DIR.exists() else []
        scripts_list = [p.name for p in SCRIPTS_DIR.iterdir()] if SCRIPTS_DIR.exists() else []
        raise FileNotFoundError(
            "No processable videos found.\n"
            f"- Looked in greenscreen: {GREENSCREEN_DIR.resolve()}\n"
            f"- Looked in scripts:    {SCRIPTS_DIR.resolve()}\n"
            f"- Greenscreen files: {gs_list}\n"
            f"- Script files:     {scripts_list}\n"
            "A video is processable if its filename contains a number (or starts with '<NUM>__') "
            "and scripts/<NUM>.txt exists."
        )

    # 3) Process all
    results=[]
    for v, s in pairs:
        log(f"📦 processing: video={v.name} script={s.name}")
        results.append(str(process_one(v, s)))

    print(json.dumps({"finals": results}, indent=2))

if __name__ == "__main__":
    main()
