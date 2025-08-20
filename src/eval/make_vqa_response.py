# -*- coding: utf-8 -*-
import json, argparse, re
from pathlib import Path

def normalize(s:str)->set:
    s = re.sub(r"[^a-z0-9 ]"," ", (s or "").lower())
    toks = [w for w in s.split() if len(w)>=2]
    return set(toks)

def score(opt:str, ctx:str)->float:
    a = normalize(opt); b = normalize(ctx)
    if not a or not b: return 0.0
    inter = len(a & b); uni = len(a | b)
    return inter/uni + 0.01*inter

def load_tokens_text(d:Path):
    return {p.stem: p.read_text(encoding="utf-8") for p in d.glob("*.txt")}

def get_list(obj, *keys):
    """从 obj 中按多个候选 key 取 list；否则返回 []"""
    for k in keys:
        v = obj.get(k)
        if isinstance(v, list): return v
    return []

def get_str(obj, *keys):
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip(): return v
    return ""

def extract_qas(item:dict):
    """尽可能鲁棒地抽取题目与选项。返回 list[dict(question, options)]"""
    qa_list = None
    for key in ("qa","questions","QAs","vqa","VQA","qa_pairs"):
        if key in item and isinstance(item[key], list):
            qa_list = item[key]; break
    if qa_list is None:  # 兜底：没有就返回长度为 0
        return []

    out = []
    for q in qa_list:
        if not isinstance(q, dict):
            out.append({"question": str(q), "options": []}); continue
        qtext = get_str(q, "question","q","text","prompt","query")
        # 选项的多种可能字段
        opts = get_list(q, "options","choices","candidates","answers","answer_options","options_list")
        if not opts:
            # A/B/C/D 分开给的
            opts = [q.get(k,"") for k in ("A","B","C","D") if k in q]
        # 清洗 & 限定 4 个
        opts = [str(x) for x in opts if isinstance(x,(str,int,float))][:4]
        out.append({"question": qtext, "options": opts})
    return out

def canonical_vid(x:str)->str:
    s = str(x or "").strip()
    s = s.split("/")[-1].split("\\")[-1]
    return s[:-4] if s.lower().endswith(".mp4") else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_txt_dir", required=True)  # 我们的 tokens 文本
    ap.add_argument("--meta_json", required=True)       # 官方标注（用来拿题目&选项）
    ap.add_argument("--out_json", required=True)
    a = ap.parse_args()

    ctx_map = load_tokens_text(Path(a.tokens_txt_dir))
    meta = json.loads(Path(a.meta_json).read_text(encoding="utf-8"))

    # 兼容 list / dict 两种结构
    items = []
    if isinstance(meta, list):
        items = meta
    elif isinstance(meta, dict):
        # dict: { "VRU_1": {...}, ... } -> 转回 list
        items = []
        for k,v in meta.items():
            if isinstance(v, dict):
                v = {"video_name": v.get("video_name") or v.get("video_path") or k, **v}
            else:
                v = {"video_name": k}
            items.append(v)
    else:
        raise TypeError("Unsupported meta structure")

    out=[]
    for item in items:
        vidstem = canonical_vid(item.get("video_name") or item.get("video_path") or item.get("video") or "")
        if not vidstem:
            continue
        ctx = ctx_map.get(vidstem, "")
        qas = extract_qas(item)
        preds=[]
        if not qas:
            # 兜底：按基准 6 类默认出 6 条（全 A），至少不缺题
            preds = [{"qid": i, "pred": "A"} for i in range(6)]
        else:
            for qi,q in enumerate(qas):
                qtext, options = q["question"], q["options"]
                text_for_match = qtext + " " + ctx
                # 没选项也兜底
                if not options:
                    preds.append({"qid": qi, "pred": "A"}); continue
                best_idx, best_s = 0, -1.0
                for idx,opt in enumerate((options + [""]*4)[:4]):  # 至少4个占位
                    s = score(opt, text_for_match)
                    if s>best_s: best_s, best_idx = s, idx
                preds.append({"qid": qi, "pred": "ABCD"[best_idx]})
        out.append({"video_name": f"{vidstem}.mp4", "qa": preds})

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", a.out_json, "videos:", len(out))

if __name__=="__main__":
    main()
