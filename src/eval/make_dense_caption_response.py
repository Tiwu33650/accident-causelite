# -*- coding: utf-8 -*-
import json, argparse, re
from pathlib import Path

def token_to_phrase(t: dict) -> str:
    txt = t.get("text","")
    if txt.startswith("EV["):
        m = re.search(r"EV\[(veh\d+)\.(\w+)\.(\w+)", txt)
        if m: return f"{m.group(1)} {m.group(3)} {m.group(2)}"
    if txt.startswith("REL["):
        m = re.search(r"REL\[(veh\d+)↔(veh\d+)\.(\w+)\.(\w+)", txt)
        if m:
            rel = {"distance":"distance","ttc":"ttc","approach":"approach","side":"side"}.get(m.group(3), m.group(3))
            return f"{m.group(1)} and {m.group(2)} {rel} {m.group(4)}"
    if txt.startswith("HAZ["):
        m = re.search(r"HAZ\[(.+?)\.(.+?)@", txt)
        if m: return f"env {m.group(1)} {m.group(2)}"
    return txt

def summarize(tokens):
    ev  = [t for t in tokens if t.get("type")=="EV"][:3]
    rel = [t for t in tokens if t.get("type")=="REL" and any(k in t.get("text","") for k in ["ttc","distance","approach"] )][:3]
    haz = [t for t in tokens if t.get("type")=="HAZ"][:2]
    phrases = [token_to_phrase(t) for t in (haz+rel+ev) if t]
    chain = " -> ".join([token_to_phrase(t) for t in (ev[:2]+rel[:1]) if t])
    parts = []
    if phrases: parts.append("; ".join(phrases))
    if chain:   parts.append(f"causal chain: {chain}")
    return ". ".join(parts) or "complex scene."

def iter_meta_generic(meta):
    """Return iterable of dicts at least having video_name/video_path."""
    if isinstance(meta, list):
        for x in meta:
            if isinstance(x, dict):
                yield x
            else:  # string
                yield {"video_name": str(x)}
    elif isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, dict):
                d = {"video_name": v.get("video_name") or v.get("video_path") or k}
                d.update(v)
                yield d
            else:  # value 非字典（可能是列表或字符串）
                yield {"video_name": k}
    else:
        raise TypeError("Unsupported meta json structure")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_dir", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--out_json", required=True)
    a = ap.parse_args()

    tokens_dir = Path(a.tokens_dir)
    token_map = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in tokens_dir.glob("*.json")}
    meta_raw = json.loads(Path(a.meta_json).read_text(encoding="utf-8"))

    out = []
    for item in iter_meta_generic(meta_raw):
        vidstem = Path(item.get("video_name") or item.get("video_path") or "").stem
        if not vidstem:  # 跳过异常条目
            continue
        toks = token_map.get(vidstem, [])
        out.append({"video_name": f"{vidstem}.mp4", "caption": summarize(toks)})

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", a.out_json, "items:", len(out))

if __name__ == "__main__":
    main()
