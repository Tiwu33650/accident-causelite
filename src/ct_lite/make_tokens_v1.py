# -*- coding: utf-8 -*-
# 文件：src/ct_lite/make_tokens_v1.py
import json, argparse, math, numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from tqdm import tqdm

@dataclass
class Token:
    type: str   # 'EV' | 'REL' | 'HAZ'
    args: Dict[str, Any]
    t0: float
    dur: float
    conf: float
    text: str

def load_tracks(tracks_json: Path):
    d = json.loads(Path(tracks_json).read_text(encoding="utf-8"))
    return d["meta"], {int(k):v for k,v in d["tracks"].items()}

def _segments_bool(mask: np.ndarray, min_len: int):
    segs = []
    i=0; n=len(mask)
    while i<n:
        if mask[i]:
            j=i
            while j<n and mask[j]: j+=1
            if (j-i)>=min_len: segs.append((i,j))
            i=j
        else:
            i+=1
    return segs

def detect_EV_for_track(tid:str, tr:List[Dict], fps:float):
    t = np.array([p["frame"] for p in tr])
    vx = np.array([p.get("vx",0) for p in tr])
    vy = np.array([p.get("vy",0) for p in tr])
    speed = np.sqrt(vx**2+vy**2)
    # 近似“纵向加速度”：对speed做差分
    acc = np.gradient(speed) * fps

    tokens = []
    # 急刹：加速度 < -A 且持续 >= 0.4s
    A = 3.0   # 像素尺度未知，先用“相对”阈值：对速度分位数自适应
    A = max(1.0, np.nanpercentile(np.abs(acc), 75))  # 自适应强度基线
    mask_brake = acc < -A
    for i,j in _segments_bool(mask_brake, min_len=int(0.4*fps)):
        t0 = float(t[i]/fps); dur = float((t[j-1]-t[i]+1)/fps)
        strength = "hard" if np.min(acc[i:j]) < -1.5*A else "mild"
        text = f"EV[veh{tid}.brake.{strength}@t={t0:.1f}s,d={dur:.1f}s]"
        tokens.append(Token("EV", {"agent": f"veh{tid}", "act":"brake","level":strength}, t0, dur, conf=0.7, text=text))

    # 急转：角速度大
    hdg = np.array([p.get("heading",0) for p in tr])
    omega = np.diff(np.unwrap(hdg)); omega = np.r_[omega[:1], omega] * fps
    W = np.nanpercentile(np.abs(omega), 80)
    mask_steer = np.abs(omega) > max(0.8, W)
    for i,j in _segments_bool(mask_steer, min_len=int(0.3*fps)):
        t0 = float(t[i]/fps); dur = float((t[j-1]-t[i]+1)/fps)
        lvl = "hard" if np.max(np.abs(omega[i:j])) > 1.5*W else "mild"
        text = f"EV[veh{tid}.steer.{lvl}@t={t0:.1f}s,d={dur:.1f}s]"
        tokens.append(Token("EV", {"agent": f"veh{tid}", "act":"steer","level":lvl}, t0, dur, conf=0.6, text=text))
    return tokens

def detect_REL_between(i:str, tr_i:List[Dict], j:str, tr_j:List[Dict], fps:float, img_scale:float=1.0):
    # 简化：按帧对齐（以i为基准）
    frames_i = [p["frame"] for p in tr_i]
    map_j = {p["frame"]:p for p in tr_j}
    pairs = [(p_i, map_j.get(p_i["frame"])) for p_i in tr_i]
    pairs = [(pi, pj) for pi,pj in pairs if pj is not None]
    if not pairs: return []

    # 序列
    rx = []; ry = []; t = []
    for pi,pj in pairs:
        rx.append(pj["cx"] - pi["cx"])
        ry.append(pj["cy"] - pi["cy"])
        t.append(pi["frame"])
    rx = np.array(rx); ry = np.array(ry); t = np.array(t)

    dist = np.sqrt(rx**2 + ry**2)
    # 接近速度（沿连线）
    drx = np.gradient(rx)*fps; dry = np.gradient(ry)*fps
    closing = -(rx*drx + ry*dry) / (np.sqrt(rx**2+ry**2)+1e-6)

    # 左右：以i的速度方向为参照
    hdg_i = np.array([p.get("heading",0) for p in tr_i if p["frame"] in t])
    side_sign = np.sign(np.sin(hdg_i) * rx - np.cos(hdg_i) * ry)  # 近似叉积符号
    side_left = side_sign > 0

    tokens=[]
    # 距离近
    thr_close = np.nanpercentile(dist, 30)  # 自适应阈值
    mask_close = dist < max(20.0*img_scale, thr_close)
    for a,b in _segments_bool(mask_close, min_len=int(0.5*fps)):
        t0 = float(t[a]/fps); dur = float((t[b-1]-t[a]+1)/fps)
        text = f"REL[veh{i}↔veh{j}.distance.close@t={t0:.1f}s,d={dur:.1f}s]"
        tokens.append(Token("REL", {"a":f"veh{i}","b":f"veh{j}","rel":"distance","level":"close"}, t0, dur, conf=0.6, text=text))

    # 接近（闭合）
    mask_closing = closing > np.nanpercentile(closing, 70)
    for a,b in _segments_bool(mask_closing, min_len=int(0.5*fps)):
        t0 = float(t[a]/fps); dur = float((t[b-1]-t[a]+1)/fps)
        text = f"REL[veh{i}↔veh{j}.approach.high@t={t0:.1f}s,d={dur:.1f}s]"
        tokens.append(Token("REL", {"a":f"veh{i}","b":f"veh{j}","rel":"approach","level":"high"}, t0, dur, conf=0.6, text=text))

    # 左/右（短片段）
    for a,b in _segments_bool(side_left, min_len=int(0.5*fps)):
        t0 = float(t[a]/fps); dur = float((t[b-1]-t[a]+1)/fps)
        text = f"REL[veh{i}↔veh{j}.side.left@t={t0:.1f}s,d={dur:.1f}s]"
        tokens.append(Token("REL", {"a":f"veh{i}","b":f"veh{j}","rel":"side","level":"left"}, t0, dur, conf=0.55, text=text))
    for a,b in _segments_bool(~side_left, min_len=int(0.5*fps)):
        t0 = float(t[a]/fps); dur = float((t[b-1]-t[a]+1)/fps)
        text = f"REL[veh{i}↔veh{j}.side.right@t={t0:.1f}s,d={dur:.1f}s]"
        tokens.append(Token("REL", {"a":f"veh{i}","b":f"veh{j}","rel":"side","level":"right"}, t0, dur, conf=0.55, text=text))

    # 粗略“接触可能”
    ttc = dist / (np.maximum(1e-3, closing))
    mask_danger = (ttc < 3.0) & np.isfinite(ttc)
    for a,b in _segments_bool(mask_danger, min_len=int(0.3*fps)):
        t0 = float(t[a]/fps); dur = float((t[b-1]-t[a]+1)/fps)
        text = f"REL[veh{i}↔veh{j}.ttc.danger@t={t0:.1f}s,d={dur:.1f}s]"
        tokens.append(Token("REL", {"a":f"veh{i}","b":f"veh{j}","rel":"ttc","level":"danger"}, t0, dur, conf=0.65, text=text))
    return tokens

def score_and_select(tokens: List[Token], fps: float, K: int = 64):
    # 简单打分：类型权重+持续时间+置信+时间覆盖多样性
    def base_score(tok: Token):
        w = {"EV":1.2, "REL":1.0, "HAZ":0.8}[tok.type]
        return w*tok.conf + 0.15*min(tok.dur, 2.0)
    toks = sorted(tokens, key=base_score, reverse=True)

    # 时间×类型分桶（4×3），保证覆盖面
    if not toks: return []
    tmax = max([tok.t0+tok.dur for tok in toks]) + 1e-6
    buckets = {(b,t):[] for b in range(4) for t in ["EV","REL","HAZ"]}
    for tok in toks:
        b = min(3, int(4*(tok.t0/tmax)))
        buckets[(b,tok.type)].append(tok)

    quota = {"EV":28, "REL":24, "HAZ":12}
    chosen=[]
    # 轮转抽取，直到到达K或桶空
    while len(chosen) < K:
        progress=False
        for t in ["EV","REL","HAZ"]:
            for b in range(4):
                if quota[t] <= 0: continue
                if buckets[(b,t)]:
                    chosen.append(buckets[(b,t)].pop(0))
                    quota[t]-=1; progress=True
                if len(chosen)>=K: break
            if len(chosen)>=K: break
        if not progress: break
    return chosen[:K]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_json", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_txt", type=str, required=True)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--topk", type=int, default=64)
    args = ap.parse_args()

    meta, tracks = load_tracks(Path(args.tracks_json))
    fps = args.fps or meta.get("fps", 6.0)

    # 1) EV
    all_tokens: List[Token] = []
    for tid, tr in tracks.items():
        all_tokens += detect_EV_for_track(str(tid), tr, fps)

    # 2) REL（仅近邻对，避免爆炸）
    ids = list(tracks.keys())
    for i_idx in range(len(ids)):
        for j_idx in range(i_idx+1, len(ids)):
            i, j = ids[i_idx], ids[j_idx]
            # 快速过滤（帧重叠且至少有若干帧）
            if len(tracks[i]) < 3 or len(tracks[j]) < 3: continue
            all_tokens += detect_REL_between(str(i), tracks[i], str(j), tracks[j], fps)

    # 3) HAZ（V1先不做或留空；可日后接入能见度/湿滑识别）
    # all_tokens += ...

    # 4) Top-K 选择
    selected = score_and_select(all_tokens, fps, K=args.topk)

    # 5) 导出
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt  = Path(args.out_txt);  out_txt.parent.mkdir(parents=True, exist_ok=True)

    json.dump([asdict(t) for t in selected], open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(out_txt, "w", encoding="utf-8") as f:
        for t in selected:
            f.write(t.text+"\n")

    print(f"[CT-Lite] tokens saved: {out_json} & {out_txt} (K={len(selected)})")

if __name__ == "__main__":
    main()
