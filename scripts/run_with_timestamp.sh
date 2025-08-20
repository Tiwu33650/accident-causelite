#!/usr/bin/env bash
set -euo pipefail

# ---------------- 基本配置（可改） ----------------
SET="${1:-MANUAL_DATA}"         # 数据子集：MANUAL_DATA / CAP_DATA / DADA_2000 / DoTA
NAME="${2:-输出}"               # 运行归档的一级目录名（当前目录下会创建 NAME/结束时间/）
PY=python                       # 你的 Python 命令（在 causelite 环境里）

# 当前项目根（用绝对路径定位“当前目录”）
PROJ="$(pwd)"

# 路径约定（输入/中间产物的默认位置）
FRAMES_DIR="$PROJ/data/processed/frames/$SET"
TRACKS_DIR="$PROJ/data/processed/tracks/$SET"
LANES_DIR="$PROJ/data/processed/lanes/$SET"   # 若没有可忽略
TOKENS_SCRIPT="$PROJ/src/ct_lite/make_tokens_v1.py"
DENSE_RESP="$PROJ/src/eval/make_dense_caption_response.py"
VQA_RESP="$PROJ/src/eval/make_vqa_response.py"
VQA_EVAL="$PROJ/src/eval/eval_vru_vqa.py"
MOD23_PY="$PROJ/src/pipeline/module23.py"

# 官方元数据（评测需要）
META_DIR="$PROJ/data/VRU-Accident/MetaData/$SET"
CAP_JSON="$META_DIR/${SET}_Dense_Caption.json"
VQA_JSON="$META_DIR/${SET}_VQA_annotation.json"

# 官方仓库路径（只做软链接，文件仍存放在本次运行目录）
VRU_MAIN="$PROJ/VRU-Accident-main"
VRU_RESP_VQA="$VRU_MAIN/Model_Response/VQA/Ours"
VRU_RESP_CAP="$VRU_MAIN/Model_Response/Dense_Captioning/Ours"

# ---------------- 运行目录（按结束时间命名） ----------------
# 先用“运行中_开始时间”写入，结束后重命名为“结束时间”
TS_START="$(date +'%Y年%m月%d日%H时%M分')"
RUN_BASE="$PROJ/$NAME"
RUN_TMP="$RUN_BASE/运行中_${TS_START}"
mkdir -p "$RUN_TMP"/{frames,tracks,tokens,responses/{VQA,Dense_Captioning},eval,module23,logs}/"$SET"

echo "[INFO] 运行开始：$TS_START"
echo "[INFO] 暂存目录：$RUN_TMP"

# ---------------- 收集/复用帧与轨迹 ----------------
# 默认用“软链接”方式把已有的帧/轨迹纳入这次运行目录。若你想复制，可把 ln -sfn 改成 rsync。
if [ -d "$FRAMES_DIR" ]; then
  ln -sfn "$FRAMES_DIR" "$RUN_TMP/frames/$SET"
else
  echo "[WARN] 未发现帧目录：$FRAMES_DIR"
fi
if [ -d "$TRACKS_DIR" ]; then
  ln -sfn "$TRACKS_DIR" "$RUN_TMP/tracks/$SET"
else
  echo "[WARN] 未发现轨迹目录：$TRACKS_DIR"
fi

# ---------------- 生成 tokens（≤64 条/视频） ----------------
if [ -f "$TOKENS_SCRIPT" ]; then
  echo "[STEP] 生成 tokens ..."
  for j in "$TRACKS_DIR"/*.json; do
    [ -e "$j" ] || continue
    vid="$(basename "$j" .json)"
    $PY "$TOKENS_SCRIPT" \
      --tracks_json "$j" \
      --out_json "$RUN_TMP/tokens/$SET/${vid}.json" \
      --out_txt  "$RUN_TMP/tokens/$SET/${vid}.txt" \
      --fps 6 --topk 64
  done |& tee "$RUN_TMP/logs/03_tokens.log"
else
  echo "[WARN] 未找到 tokens 脚本：$TOKENS_SCRIPT"
fi

# ---------------- 生成 Model_Response（Dense Caption & VQA） ----------------
echo "[STEP] 生成 Dense Caption 响应 ..."
$PY "$DENSE_RESP" \
  --tokens_dir "$RUN_TMP/tokens/$SET" \
  --meta_json  "$CAP_JSON" \
  --out_json   "$RUN_TMP/responses/Dense_Captioning/${SET}_Dense_Captioning_response.json" \
  |& tee "$RUN_TMP/logs/04_caption.log"

echo "[STEP] 生成 VQA 响应 ..."
$PY "$VQA_RESP" \
  --tokens_txt_dir "$RUN_TMP/tokens/$SET" \
  --meta_json      "$VQA_JSON" \
  --out_json       "$RUN_TMP/responses/VQA/${SET}_VQA_response.json" \
  |& tee "$RUN_TMP/logs/05_vqa.log"

# ---------------- 本地评测（VQA） ----------------
echo "[STEP] 本地评测 VQA ..."
$PY "$VQA_EVAL" \
  --gt_json   "$VQA_JSON" \
  --pred_json "$RUN_TMP/responses/VQA/${SET}_VQA_response.json" \
  --save_json "$RUN_TMP/eval/${SET}_VQA_acc.json" \
  |& tee "$RUN_TMP/logs/06_eval_vqa.log"

# ---------------- 模块2+3（空间锚 & 小改一项反事实） ----------------
if [ -f "$MOD23_PY" ]; then
  echo "[STEP] 模块2+3 输出 ..."
  $PY "$MOD23_PY" \
    --tokens_dir "$RUN_TMP/tokens/$SET" \
    --tracks_dir "$TRACKS_DIR" \
    --out_dir    "$RUN_TMP/module23/$SET" \
    |& tee "$RUN_TMP/logs/07_mod23.log"
else
  echo "[WARN] 未找到模块2+3脚本：$MOD23_PY"
fi

# ---------------- 将运行目录重命名为“结束时间” ----------------
TS_END="$(date +'%Y年%m月%d日%H时%M分')"
RUN_FINAL="$RUN_BASE/$TS_END"
mv "$RUN_TMP" "$RUN_FINAL"
ln -sfn "$RUN_FINAL" "$RUN_BASE/latest"

echo "[INFO] 运行结束：$TS_END"
echo "[INFO] 本次结果目录：$RUN_FINAL"
echo "$TS_START -> $TS_END" > "$RUN_FINAL/运行时间.txt"

# ---------------- 官方目录软链接（评测/对比用） ----------------
mkdir -p "$VRU_RESP_VQA" "$VRU_RESP_CAP"
ln -sfn "$RUN_FINAL/responses/VQA/${SET}_VQA_response.json" \
       "$VRU_RESP_VQA/${SET}_VQA_response.json"
ln -sfn "$RUN_FINAL/responses/Dense_Captioning/${SET}_Dense_Captioning_response.json" \
       "$VRU_RESP_CAP/${SET}_Dense_Captioning_response.json"

# ---------------- 结果摘要 ----------------
echo "--------------------------------------------------"
echo "[DONE] 归档完成。关键信息："
echo "  运行目录：$RUN_FINAL"
echo "  VQA响应： $RUN_FINAL/responses/VQA/${SET}_VQA_response.json"
echo "  CAP响应： $RUN_FINAL/responses/Dense_Captioning/${SET}_Dense_Captioning_response.json"
echo "  VQA评测： $RUN_FINAL/eval/${SET}_VQA_acc.json"
echo "  模块2+3： $RUN_FINAL/module23/$SET/{chains.json,counterfactuals.json}"
echo "  最新指针：$RUN_BASE/latest -> $(readlink -f "$RUN_BASE/latest")"
echo "  官方链接：$VRU_RESP_VQA/${SET}_VQA_response.json"
echo "          $VRU_RESP_CAP/${SET}_Dense_Captioning_response.json"
echo "--------------------------------------------------"
