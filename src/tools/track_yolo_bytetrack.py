# -*- coding: utf-8 -*-
# 文件：src/tools/track_yolo_bytetrack.py
from pathlib import Path
import argparse, json, numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def centers_from_xyxy(xyxy):
    x1,y1,x2,y2 = xyxy
    return float((x1+x2)/2), float((y1+y2)/2), float(x2-x1), float(y2-y1)

def compute_vel_heading(tr, fps):
    # ---- 短轨迹兜底：长度<2，直接置零并返回 ----
    if len(tr) < 2:
        for p in tr:
            p["vx"] = 0.0
            p["vy"] = 0.0
            p["speed"] = 0.0
            p["heading"] = 0.0
        return
    # 根据连续帧中心点，计算 vx, vy, speed, heading（弧度）
    xs = np.array([p["cx"] for p in tr]); ys = np.array([p["cy"] for p in tr])
    vx = np.gradient(xs) * fps; vy = np.gradient(ys) * fps
    spd = np.sqrt(vx**2 + vy**2)
    hdg = np.arctan2(vy, vx)  # [-pi,pi]
    for i,p in enumerate(tr):
        p["vx"], p["vy"], p["speed"], p["heading"] = float(vx[i]), float(vy[i]), float(spd[i]), float(hdg[i])

def track_one(frames_dir: Path, model: YOLO, out_json: Path, device="cuda", conf=0.25, iou=0.5):
    results = model.track(
        source=str(frames_dir),
        stream=True,
        tracker="/data/SuiBowen/accident-causelite/models/cfg/trackers/bytetrack.yaml",
        conf=conf, iou=iou, device=device, verbose=False
    )
    tracks = {}  # id -> list of dict per frame
    frame_idx = 0
    for r in tqdm(results, desc=f"Track {frames_dir.name}"):
        frame_idx += 1
        if r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.cpu().numpy().astype(int)
        xyxys = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for i, tid in enumerate(ids):
            cx, cy, w, h = centers_from_xyxy(xyxys[i])
            item = {
                "frame": frame_idx,
                "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
                "conf": float(confs[i]), "cls": int(clss[i])
            }
            tracks.setdefault(int(tid), []).append(item)

    # 估计 fps（按抽帧时约定 6FPS）
    fps = 6.0
    # 为每条轨迹计算速度/朝向（带兜底）
    for tid, tr in tracks.items():
        tr.sort(key=lambda x: x["frame"])
        compute_vel_heading(tr, fps)

    # 过滤掉只有 1 帧的短轨迹（避免后续模块噪声）
    filtered = {tid: tr for tid, tr in tracks.items() if len(tr) >= 2}

    meta = {"video_id": frames_dir.name, "fps": fps, "num_frames": frame_idx}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "tracks": filtered}, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", type=str, required=True, help="抽帧目录根，比如 data/processed/frames")
    ap.add_argument("--out_root", type=str, required=True, help="tracks 输出根，比如 data/processed/tracks")
    ap.add_argument("--model", type=str, default="yolov8n.pt")  # 先n/s就行，后续可换强模型
    ap.add_argument("--device", type=str, default="cuda")       # 或 'cpu'
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    frames_root = Path(args.frames_root); out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    for frames_dir in sorted(frames_root.iterdir()):
        if not frames_dir.is_dir(): continue
        out_json = out_root / f"{frames_dir.name}.json"
        track_one(frames_dir, model, out_json, device=args.device, conf=args.conf, iou=args.iou)

if __name__ == "__main__":
    main()
