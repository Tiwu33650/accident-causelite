# -*- coding: utf-8 -*-
import json, argparse, re
from pathlib import Path
from collections import defaultdict, deque

LETTER = "ABCD"

def canonical_vid(x:str)->str:
    s = str(x or "").strip().replace("\\","/")
    s = s.split("/")[-1]
    return s[:-4] if s.lower().endswith(".mp4") else s

def load_pred(path:Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    pred = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict): continue
            vid = canonical_vid(item.get("video_name") or item.get("video_path") or item.get("video") or "")
            q = item.get("qa") or item.get("pred") or []
            if isinstance(q, dict):
                q = [q[str(i)] if str(i) in q else q.get(i,"") for i in range(len(q))]
            qq=[]
            for x in (q or []):
                p = x.get("pred") if isinstance(x, dict) else x
                if p: qq.append(str(p).strip()[:1].upper())
            if vid: pred[vid]=qq
    elif isinstance(data, dict):
        for k,v in data.items():
            vid = canonical_vid(k)
            q = v.get("qa") if isinstance(v, dict) else v
            if isinstance(q, dict):
                q = [q[str(i)] if str(i) in q else q.get(i,"") for i in range(len(q))]
            qq=[]
            for x in (q or []):
                p = x.get("pred") if isinstance(x, dict) else x
                if p: qq.append(str(p).strip()[:1].upper())
            if vid: pred[vid]=qq
    else:
        raise ValueError("Unexpected pred json structure.")
    return pred

# -------------- GT 解析（任意嵌套） --------------
def _looks_like_qnode(d):
    if not isinstance(d, dict): return False
    keys = set(k.lower() for k in d.keys())
    # 直接命中
    if {"question","options"} <= keys: return True
    if {"q","choices"} <= keys: return True
    if {"question","a","b","c","d"} <= keys: return True
    if {"q_text"} & keys: return True
    # 带答案/索引/文本的条目
    if {"answer","answer_idx","answer_id","correct","label","gt","answer_text","correct_idx"} & keys:
        return True
    # option1..4 / choice1..4 / option_1..4
    if any(re.fullmatch(r'(option|choice)[ _]?[1-4]', k) for k in keys):
        return True
    return False

def _flatten_qnodes(container):
    out=[]
    dq=deque([container])
    while dq:
        x=dq.popleft()
        if isinstance(x, dict):
            if _looks_like_qnode(x):
                out.append(x); continue
            dq.extend(x.values())
        elif isinstance(x, list):
            dq.extend(x)
    return out

def iter_gt_items(meta):
    if isinstance(meta, list):
        for it in meta:
            if isinstance(it, dict):
                vid = canonical_vid(it.get("video_name") or it.get("video_path") or it.get("name") or it.get("video") or "")
                qnodes = _flatten_qnodes(it.get("qa") or it.get("questions") or it.get("QAs") or it.get("VQA") or it.get("vqa") or it)
                yield vid, qnodes
            else:
                yield canonical_vid(str(it)), []
    elif isinstance(meta, dict):
        for k,v in meta.items():
            if isinstance(v, dict):
                vid = canonical_vid(v.get("video_name") or v.get("video_path") or v.get("name") or v.get("video") or k)
                qnodes = _flatten_qnodes(v.get("qa") or v.get("questions") or v.get("QAs") or v.get("VQA") or v.get("vqa") or v)
                yield vid, qnodes
            else:
                yield canonical_vid(k), []
    else:
        raise TypeError("Unsupported GT structure")

def _first_str(*vals):
    for t in vals:
        if isinstance(t, str) and t.strip(): return t.strip()
    return None

def _first_int(*vals):
    for t in vals:
        if isinstance(t, int): return t
        if isinstance(t, str) and t.strip().isdigit(): return int(t.strip())
    return None

def _get_options(q):
    """大小写无关地取选项；支持 options/choices/answers… 或 A/B/C/D、option1..4 等"""
    ql = {k.lower(): v for k, v in q.items()}  # 键全部小写
    # 1) 数组形式
    for key in ("options","choices","answers","candidates","option_list","options_list"):
        v = ql.get(key)
        if isinstance(v, list) and v:
            return [str(x) for x in v[:4]]
    # 2) A/B/C/D 分散
    opts = []
    for k in ("a","b","c","d"):
        if k in ql:
            opts.append(str(ql[k]))
    if len(opts) >= 2:
        return opts[:4]
    # 3) option1..4 / choice1..4 / option_1..4
    col=[]
    for i in range(1,5):
        for base in ("option","choice"):
            for sep in ("", " ", "_"):
                k = f"{base}{sep}{i}"
                if k in ql:
                    col.append(str(ql[k]))
    if col:
        return col[:4]
    return []

def _ans_letter_from_item(q):
    """大小写无关地还原 GT 为 A/B/C/D；支持字母、索引、文本匹配选项"""
    ql = {k.lower(): v for k, v in q.items()}  # 键全部小写
    LETTER = "ABCD"

    def _first_str(*vals):
        for t in vals:
            if isinstance(t, str) and t.strip():
                return t.strip()
        return None

    def _first_int(*vals):
        for t in vals:
            if isinstance(t, int):
                return t
            if isinstance(t, str) and t.strip().isdigit():
                return int(t.strip())
        return None

    # 1) 直接是字母/字符串（含 "GT" 大写 → 转成小写键后就是 "gt"）
    s = _first_str(ql.get("answer"), ql.get("gt"), ql.get("label"), ql.get("correct"),
                   ql.get("answer_label"), ql.get("correct_label"), ql.get("correct_option"),
                   ql.get("ans"), ql.get("gt_label"))
    if s:
        ch = s[:1].upper()
        if ch in LETTER:
            return ch
        m = re.search(r'(\d+)', s)
        if m:
            return LETTER[int(m.group(1)) % 4]

    # 2) 索引（0~3 或 1~4）
    idx = _first_int(ql.get("answer_idx"), ql.get("answer_id"), ql.get("index"),
                     ql.get("label_id"), ql.get("gt_idx"), ql.get("correct_idx"),
                     ql.get("option"), ql.get("option_index"))
    if idx is not None:
        if 1 <= idx <= 4:
            idx -= 1
        return LETTER[idx % 4]

    # 3) 文本匹配到选项
    opts = _get_options(q)
    ans_text = _first_str(ql.get("answer_text"), ql.get("gt_text"),
                          ql.get("correct_text"), ql.get("answer_str"))
    if ans_text and opts:
        lo = [o.strip().lower() for o in opts]
        try:
            j = lo.index(ans_text.strip().lower())
            return LETTER[j % 4]
        except ValueError:
            pass

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", required=True)
    ap.add_argument("--pred_json", required=True)
    ap.add_argument("--save_json", required=True)
    ap.add_argument("--peek", action="store_true")
    a = ap.parse_args()

    gt_raw  = json.loads(Path(a.gt_json).read_text(encoding="utf-8"))
    predmap = load_pred(Path(a.pred_json))

    if a.peek:
        # 打印样本对齐信息与前几个题目键
        vids = []
        for vid, qnodes in iter_gt_items(gt_raw):
            if vid: vids.append((vid, len(qnodes)))
            if len(vids)>=10: break
        print("GT sample vids & qcount:", vids)
        print("PRED sample vids:", list(predmap.keys())[:10])

    tot_q = cor_q = 0
    cats = defaultdict(lambda: {"n":0,"c":0})
    miss_videos=set(); len_mismatch=set()

    for vid, qnodes in iter_gt_items(gt_raw):
        preds = predmap.get(vid, [])
        if not qnodes:
            continue
        if not preds:
            miss_videos.add(vid);
            continue
        for qi, q in enumerate(qnodes):
            if not isinstance(q, dict): continue
            cat = (q.get("category") or q.get("type") or "Unknown").strip()
            gtL = _ans_letter_from_item(q)
            if gtL is None:
                continue
            if qi >= len(preds):
                len_mismatch.add(vid); break
            prL = str(preds[qi]).strip()[:1].upper()
            cats[cat]["n"] += 1; tot_q += 1
            if prL == gtL:
                cats[cat]["c"] += 1; cor_q += 1

    res = {
        "total_questions": tot_q,
        "total_correct": cor_q,
        "overall_acc": round(cor_q / tot_q, 4) if tot_q else 0.0,
        "by_category": {k: {"n": v["n"], "c": v["c"], "acc": round(v["c"]/v["n"],4) if v["n"] else 0.0}
                        for k,v in cats.items()},
        "num_missing_videos": len(miss_videos),
        "num_length_mismatch": len(len_mismatch),
    }
    Path(a.save_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.save_json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", a.save_json)
    print("Overall Acc:", res["overall_acc"])
    print("Total Q:", res["total_questions"])
    if miss_videos: print("Missing videos (sample):", list(sorted(miss_videos))[:10])

if __name__ == "__main__":
    main()
