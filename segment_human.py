#!/usr/bin/env python3
import os, sys, subprocess, shlex
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not installed. Run: pip install mediapipe==0.10.14")
    sys.exit(1)

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("ERROR: ffmpeg not found on PATH. Install it (e.g., sudo apt-get install -y ffmpeg).")
        sys.exit(1)

def fast_transparent_segment(
    input_path: str,
    output_mov: str,
    target_height: int = 720,
    threshold: float = 0.50,
    blur_ksize: int = 21,
    alpha_soften: float = 0.08,   # soften edges (0..0.3). Higher = softer
):
    """
    Fastest CPU path to transparent video:
      - MediaPipe segmentation on downscaled frames
      - Streams RGBA frames to ffmpeg → ProRes 4444 MOV with alpha
    """
    check_ffmpeg()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w, src_h = int(cap.get(3)), int(cap.get(4))

    # Keep aspect ratio, scale by height
    if target_height and target_height < src_h:
        scale = target_height / src_h
        W, H = int(round(src_w * scale)), int(round(src_h * scale))
    else:
        W, H = src_w, src_h

    # Launch ffmpeg writer for RGBA → ProRes 4444 with alpha
    ffmpeg_cmd = f"""
        ffmpeg -y
        -f rawvideo -pix_fmt rgba -s {W}x{H} -r {fps:.02f} -i -
        -an
        -c:v prores_ks -profile:v 4 -pix_fmt yuva444p10le
        -movflags +faststart
        "{output_mov}"
    """
    proc = subprocess.Popen(
        shlex.split(ffmpeg_cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=10**7
    )

    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Pre-build kernels for a tiny morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if (W, H) != (src_w, src_h):
                frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_AREA)

            # MediaPipe expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Segmentation mask in [0..1]
            mask = mp_selfie.process(frame_rgb).segmentation_mask

            # Edge refinement: blur + tiny morph close to remove holes
            if blur_ksize > 0:
                mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Convert soft mask to alpha with a gentle soft-threshold
            # push threshold to reduce halos, then remap to [0..255]
            hard = (mask - threshold) / max(1e-6, (1.0 - threshold))
            soft = np.clip(hard, 0, 1)

            if alpha_soften > 0:
                # Slight extra blur for nicer hairlines (cheap)
                k = max(3, int(blur_ksize * alpha_soften) | 1)
                soft = cv2.GaussianBlur(soft, (k, k), 0)

            alpha = (soft * 255.0).astype(np.uint8)

            # Compose RGBA: use the original (RGB) as color, alpha from mask
            rgba = np.dstack((frame_rgb, alpha))

            # Write raw RGBA to ffmpeg
            proc.stdin.write(rgba.tobytes())

            frame_idx += 1
            if frame_idx % 100 == 0:
                sys.stdout.write(f"\rProcessed {frame_idx} frames…")
                sys.stdout.flush()

    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        _, err = proc.communicate()
        if proc.returncode != 0:
            # Print ffmpeg error for quick diagnosis
            if err:
                sys.stderr.write(err.decode("utf-8", errors="ignore"))
            raise RuntimeError("ffmpeg failed while writing the transparent MOV.")

    print(f"\n✅ Done. Transparent output → {output_mov}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fast_transparent_segment.py input.mp4 output.mov [target_height]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 720

    fast_transparent_segment(
        input_path,
        output_path,
        target_height=height,
        threshold=0.50,     # raise to 0.55–0.60 for less halo
        blur_ksize=21,      # reduce to 15 for speed, increase for smoother edges
        alpha_soften=0.08   # 0.0 disables extra feathering
    )
