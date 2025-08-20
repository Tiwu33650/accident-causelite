# -*- coding: utf-8 -*-
# 文件：src/tools/extract_frames.py
import cv2, os, math, argparse
from pathlib import Path
from tqdm import tqdm

def extract_one(video_path: Path, out_dir: Path, fps_target: float = 6.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps_src / fps_target)))
    idx = 0; saved = 0
    pbar = tqdm(total=total, desc=f"Extract {video_path.name} -> {out_dir}")
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            saved += 1
            out_path = out_dir / f"{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
        idx += 1; pbar.update(1)
    cap.release(); pbar.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None, help="单个视频路径")
    ap.add_argument("--video_dir", type=str, default=None, help="视频目录（批量）")
    ap.add_argument("--out_root", type=str, required=True, help="输出帧目录根，比如 data/processed/frames")
    ap.add_argument("--fps", type=float, default=6.0)
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    if args.video:
        vid = Path(args.video)
        vid_id = vid.stem
        extract_one(vid, out_root / vid_id, args.fps)
    else:
        vdir = Path(args.video_dir)
        for vid in sorted(vdir.glob("*.*")):
            if vid.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv"]: continue
            extract_one(vid, out_root / vid.stem, args.fps)

if __name__ == "__main__":
    main()
