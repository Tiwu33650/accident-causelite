# -*- coding: utf-8 -*-
import json, math, argparse
from pathlib import Path

# ---------- 工具 ----------
def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def dump_json(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def last_valid(tr):  # 取接近事故的末端状态（轨迹已按 frame 排好）
    return tr[-1] if tr else None

def dist(p, q):
    return math.hypot((p["cx"]-q["cx"]), (p["cy"]-q["cy"]))

# ---------- 模块2：空间锚 ----------
def build_spatial_anchors(tracks_json, lane_json=None):
    """给定跟踪结果，产出 left/right、same-lane、contact 三类锚 + 置信度"""
    data = load_json(tracks_json)
    trmap = data.get("tracks", {})
    # 末端帧的 pair-wise 关系
    tids = sorted(trmap.keys(), key=lambda k:int(k))
    tips = {tid: last_valid(sorted(trmap[tid], key=lambda x:x["frame"])) for tid in tids}
    anchors = {"LR": [], "LANE": [], "CONTACT": []}

    # 1) 左/右（以目标对象A为参照，看B在其车头坐标系的左右）
    for a in tids:
        pa = tips[a]
        if not pa: continue
        # 朝向向量（vx, vy）；模小就用上一帧差分已在跟踪里算过
        vx, vy = pa.get("vx",0.0), pa.get("vy",0.0)
        if abs(vx)+abs(vy) < 1e-3: vx,vy = 1.0,0.0  # 没速度时用默认朝向
        for b in tids:
            if a==b: continue
            pb = tips[b];
            if not pb: continue
            # 计算 B 在 A 的横向符号：sign(cross([vx,vy],[bx-ax,by-ay]))
            dx, dy = pb["cx"]-pa["cx"], pb["cy"]-pa["cy"]
            cross = vx*dy - vy*dx
            side = "left" if cross>0 else "right"
            conf = min(1.0, max(0.2, abs(cross)/(abs(vx)+abs(vy)+1e-6)))
            anchors["LR"].append({"ref": int(a), "oth": int(b), "side": side, "conf": round(conf,3)})

    # 2) 同/异车道（若无车道线，近似用“纵向方向相似 + 横向间距小”判同车道）
    # lane_json 可留空；先给近似逻辑
    for a in tids:
        pa = tips[a];
        if not pa: continue
        for b in tids:
            if a>=b: continue
            pb = tips[b];
            if not pb: continue
            # 横向距离阈值（像素域粗阈）
            same = abs(pa["cx"]-pb["cx"]) < 60
            anchors["LANE"].append({"pair":[int(a),int(b)], "same_lane": bool(same), "conf": 0.6 if same else 0.5})

    # 3) 接触（末端距离 + 接近速度趋势）
    fps = (data.get("meta") or {}).get("fps", 6.0)
    for a in tids:
        pa = tips[a];
        if not pa: continue
        for b in tids:
            if a>=b: continue
            pb = tips[b];
            if not pb: continue
            d = dist(pa, pb)
            # 末端 d 小 且 两者速度向量夹角朝向靠拢 → contact
            va = (pa.get("vx",0.0), pa.get("vy",0.0))
            vb = (pb.get("vx",0.0), pb.get("vy",0.0))
            dot = va[0]*vb[0] + va[1]*vb[1]
            mag = (math.hypot(*va)*math.hypot(*vb) + 1e-6)
            cos = dot / mag
            contact = (d < 40) and (cos < 0)  # 相向且距离小
            conf = 0.7 if contact else 0.4 if d<60 else 0.2
            anchors["CONTACT"].append({"pair":[int(a),int(b)], "contact": bool(contact), "conf": round(conf,3), "d": round(d,1)})

    return anchors

def posthoc_consistency_edit(tokens, anchors):
    """事后审校：若便签里“左/右、同/异车道、接触”与先验冲突且先验置信度高，则替换关键词并标注"""
    txts = [t.get("text","") for t in tokens]
    notes = []
    def repl_side(s):
        # EV/REL token 文本中的"...left..."或"...right..."替换
        if "left" in s and "right" not in s:
            return s.replace("left","right")
        if "right" in s and "left" not in s:
            return s.replace("right","left")
        return s

    # 左/右
    for a in anchors.get("LR", []):
        if a["conf"] < 0.6: continue
        ref, oth, side = a["ref"], a["oth"], a["side"]
        patt1 = f"REL[veh{ref}↔veh{oth}."
        patt2 = f"EV[veh{oth}."
        for i,s in enumerate(txts):
            if side=="left" and (patt1 in s or patt2 in s) and "right" in s:
                notes.append(f"flip right->left for veh{ref} vs veh{oth}")
                txts[i] = repl_side(s)
            if side=="right" and (patt1 in s or patt2 in s) and "left" in s:
                notes.append(f"flip left->right for veh{ref} vs veh{oth}")
                txts[i] = repl_side(s)

    # 同/异车道（仅在明确写了 same/other lane 的便签上修）
    for a in anchors.get("LANE", []):
        if a["conf"] < 0.55: continue
        p = a["pair"]; same = a["same_lane"]
        patt = f"REL[veh{p[0]}↔veh{p[1]}."
        for i,s in enumerate(txts):
            if patt in s:
                if same and "different_lane" in s:
                    txts[i] = s.replace("different_lane","same_lane"); notes.append("fix lane different->same")
                if (not same) and "same_lane" in s:
                    txts[i] = s.replace("same_lane","different_lane"); notes.append("fix lane same->different")

    # 接触（把明显相悖的 no_contact/contact 改过来）
    for a in anchors.get("CONTACT", []):
        if a["conf"] < 0.65: continue
        p = a["pair"]; c = a["contact"]
        patt = f"REL[veh{p[0]}↔veh{p[1]}."
        for i,s in enumerate(txts):
            if patt in s:
                if c and "no_contact" in s:
                    txts[i] = s.replace("no_contact","contact"); notes.append("fix no_contact->contact")
                if (not c) and ".contact@" in s and "no_contact" not in s:
                    txts[i] = s.replace(".contact@",".no_contact@"); notes.append("fix contact->no_contact")

    # 生成新 tokens 列表
    new_tokens = []
    for i,t in enumerate(tokens):
        nt = dict(t)
        nt["text"] = txts[i]
        new_tokens.append(nt)
    return new_tokens, notes

# ---------- 模块3：小改一项反事实 ----------
CF_RULES = [
    # (检测关键词, 改成的文本片段, 解释)
    ("REL[", ("distance.close","distance.far"), "把距离变远"),        # 近->远
    ("REL[", ("ttc.low","ttc.safe"), "把TTC变安全"),                 # 低TTC->安全
    ("HAZ[road_wet", ("road_wet.high","road_wet.none"), "路不湿"),   # 路滑->不滑
    ("EV[", ("speeding","slow"), "减速"),                            # 超速->慢
]

def make_counterfactuals(token_texts, max_per_video=2):
    out=[]
    for s in token_texts:
        for key, (a,b), why in CF_RULES:
            if key in s and a in s:
                out.append({"edit_from": a, "edit_to": b, "reason": why, "orig": s, "edited": s.replace(a,b)})
                if len(out)>=max_per_video: return out
    return out

# ---------- 语言模型（7B） ----------
def call_lm(prompt, model_path=None, max_new_tokens=180):
    """
    若提供 model_path，使用 transformers 加载本地 7B；否则用非常轻的规则兜底，先跑通流程。
    你准备好 7B 后，把 --model_path 指到模型目录即可。
    """
    if model_path and Path(model_path).exists():
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
            pipe = TextGenerationPipeline(model=mdl, tokenizer=tok)
            out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
            return out.split(prompt,1)[-1].strip()
        except Exception as e:
            return f"[LM-ERROR:{e}]"
    # 兜底生成：从便签凑3-5步
    lines = [x.strip() for x in prompt.splitlines() if x.strip().startswith(("EV[","REL[","HAZ["))]
    steps = []
    for i,s in enumerate(lines[:5]):
        if "ttc.low" in s or "distance.close" in s: steps.append(f"{i+1}. 关键危险接近，碰撞风险升高")
        elif "road_wet" in s: steps.append(f"{i+1}. 路面湿滑，制动距离变长")
        elif "brake" in s or "turn" in s: steps.append(f"{i+1}. 车辆急刹/转向引发不稳定")
        else: steps.append(f"{i+1}. 重要因素：{s[:60]}...")
    return "\n".join(steps[:5])

def make_chain_prompt(token_texts):
    head = "这是逐步的因果摘要：\n" + "\n".join(token_texts[:64]) + \
           "\n请用3–5步写出因果链，禁止臆造，不确定处请省略。"
    return head

def make_cf_prompt(token_texts, edited):
    tpl = (
        "原始便签：\n{orig}\n\n"
        "编辑后便签：\n{edt}\n\n"
        "问：如果把{factor_from}改为{factor_to}，是否还会发生事故？"
        "请回答“是/否+一句话理由+自信度0~1”。"
    )
    return tpl.format(
        orig="\n".join(token_texts[:64]),
        edt="\n".join([edited["edited"]]),
        factor_from=edited["edit_from"], factor_to=edited["edit_to"]
    )

# ---------- 主流程 ----------
def run_for_split(tokens_dir, tracks_dir, out_dir, model_path=None, lanes_dir=None, max_cf=2):
    tokens_dir = Path(tokens_dir); tracks_dir=Path(tracks_dir); out_dir=Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_out = []
    cf_out = []
    vids = sorted([p.stem for p in tokens_dir.glob("*.json")])

    for vid in vids:
        toks = load_json(tokens_dir/f"{vid}.json")
        tok_texts = [t.get("text","") for t in toks]

        anchors = build_spatial_anchors(tracks_dir/f"{vid}.json", None)
        toks_fixed, notes = posthoc_consistency_edit(toks, anchors)
        tok_texts_fixed = [t.get("text","") for t in toks_fixed]

        # 因果链
        chain = call_lm(make_chain_prompt(tok_texts_fixed), model_path=model_path)
        chain_out.append({"video_name": f"{vid}.mp4", "chain": chain, "notes": notes, "anchors": anchors})

        # 反事实（小改一项）
        cfs = make_counterfactuals(tok_texts_fixed, max_per_video=max_cf)
        cf_res=[]
        for cf in cfs:
            ans = call_lm(make_cf_prompt(tok_texts_fixed, cf), model_path=model_path, max_new_tokens=80)
            cf_res.append({"edit": cf, "answer": ans})
        cf_out.append({"video_name": f"{vid}.mp4", "counterfactuals": cf_res})

    dump_json(chain_out, out_dir/"chains.json")
    dump_json(cf_out, out_dir/"counterfactuals.json")
    print("saved:", out_dir/"chains.json", out_dir/"counterfactuals.json")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_dir", required=True)
    ap.add_argument("--tracks_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", default=None, help="本地7B模型目录（HuggingFace格式）")
    a = ap.parse_args()
    run_for_split(a.tokens_dir, a.tracks_dir, a.out_dir, model_path=a.model_path)
