#!/home/joao/miniconda3/envs/whisper_env/bin/python
import os, gc, sys
from pathlib import Path
import whisper
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import json


def write_srt(segments, path, time_offset=0.0):
    def fmt(t):
        h, m, s = int(t // 3600), int((t % 3600) // 60), int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = fmt(seg["start"] + time_offset)
            end = fmt(seg["end"] + time_offset)
            text = seg["text"].strip().replace("-->", "→")
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def split_video_minimal(video_path, output_dir="output_split", silence_seconds=5, model_size="small"):
    os.makedirs(output_dir, exist_ok=True)
    base = Path(video_path).stem

    print("🗣️ Transcribing with Whisper...")
    model = whisper.load_model(model_size)
    result = model.transcribe(video_path)
    segments = result["segments"]
    last_seg = segments[-1]
    outro_start = float(last_seg["start"])
    print(f"🕒 Outro starts at {outro_start:.2f}s")

    padded_srt_path = Path(output_dir) / f"{base}_padded.srt"
    write_srt(segments, padded_srt_path, silence_seconds)

    clip = VideoFileClip(video_path)
    intro_clip = clip.subclip(0, outro_start).without_audio()
    outro_clip = clip.subclip(outro_start).without_audio()

    intro_path = Path(output_dir) / f"{base}_intro_muted.mp4"
    outro_path = Path(output_dir) / f"{base}_outro_muted.mp4"
    audio_path = Path(output_dir) / f"{base}_padded.mp3"

    intro_clip.write_videofile(str(intro_path), codec="libx264", audio=False, verbose=False, logger=None)
    outro_clip.write_videofile(str(outro_path), codec="libx264", audio=False, verbose=False, logger=None)

    # save padded mp3
    temp_wav = Path(output_dir) / f"{base}_temp.wav"
    clip.audio.write_audiofile(str(temp_wav), fps=44100, logger=None)
    silence = AudioSegment.silent(duration=silence_seconds * 1000)
    padded = silence + AudioSegment.from_wav(temp_wav)
    padded.export(audio_path, format="mp3")
    os.remove(temp_wav)

    clip.close()
    intro_clip.close()
    outro_clip.close()
    gc.collect()

    outputs = {
        "intro_video": str(intro_path),
        "outro_video": str(outro_path),
        "padded_audio": str(audio_path),
        "padded_srt": str(padded_srt_path),
        "outro_start": outro_start,
        "last_sentence": segments[-1]["text"]
    }

    out_json = Path(output_dir) / f"{base}_metadata.json"
    json.dump(outputs, open(out_json, "w"), indent=2)
    print(f"✅ Whisper split done → {out_json}")
    return outputs


if __name__ == "__main__":
    video_path = "greenscreen/1.mp4"  # hardcoded or sys.argv[1]
    outputs = split_video_minimal(video_path)

    # save outputs to JSON for main.py to read
    meta_path = Path(__file__).resolve().parent / "output_split" / "input_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"✅ Metadata saved → {meta_path}")
