#!/bin/bash
set -euo pipefail

# ===== User Variables =====
SCENE=reichstag
IMB=/home/ach17765lb/image-matching-benchmark
JOUT=/home/ach17765lb/JamMa/outputs
SUB=/home/ach17765lb/JamMa/imc_submission
DATA=/home/ach17765lb/data

# CPU系ライブラリのスレッドを固定（暴走防止）
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "[1] クリーンアップ"
rm -rf "${IMB}/jobs"/* || true
rm -rf "${IMB}/results/phototourism/${SCENE}/jamma-kp_2048_jamma-desc" || true
rm -rf "${IMB}/packed-val/phototourism/${SCENE}" || true

echo "[2] val split を ${SCENE} のみに固定"
mkdir -p "${IMB}/json/data"
python3 - <<'PY'
import json, pathlib
p=pathlib.Path("/home/ach17765lb/image-matching-benchmark/json/data/phototourism_val.json")
p.parent.mkdir(parents=True, exist_ok=True)
json.dump(["reichstag"], p.open("w"))
print("[ok] wrote", p)
PY

echo "[3] features/matches を results にインポート（IMC 形式 → results/ にコピー）"
python3 "${IMB}/import_features.py" \
  --datasets phototourism --subset val \
  --path_features "${SUB}" \
  --path_results  "${IMB}/results" \
  --kp_name jamma_kp --desc_name jamma_desc --match_name jamma_match

echo "[3.1] config_common.match を保険で削除（schema適合）"
python3 - <<'PY'
import json, pathlib
p=pathlib.Path("/home/ach17765lb/JamMa/imc_submission/config_jamma.json")
cfg=json.loads(p.read_text())
cfg[0]["config_common"].pop("match", None)
p.write_text(json.dumps(cfg, indent=2))
print("[ok] config_common.match removed (if existed)")
PY

echo "[3.2] matches_inlier のインデックス範囲外カラムを除去（sanitize）"
python3 - <<'PY'
import h5py, numpy as np, pathlib
SCENE="reichstag"
RES_ROOT = pathlib.Path("/home/ach17765lb/image-matching-benchmark/results/phototourism")/SCENE/"jamma-kp_2048_jamma-desc"
KP_H5    = pathlib.Path("/home/ach17765lb/JamMa/imc_submission/phototourism")/SCENE/"keypoints.h5"
targets = [
    RES_ROOT/"jamma-match/no_filter/matches_inlier.h5",
    RES_ROOT/"jamma-match/no_filter/matches_imported_stereo_0.h5",
]
with h5py.File(KP_H5,"r") as fkp:
    kp_len = {k: fkp[k].shape[0] for k in fkp.keys()}
def sanitize(path):
    if not path.exists():
        print(f"[skip] {path} (not found)"); return
    fixed = dropped = 0
    with h5py.File(path,"a") as fm:
        for k in list(fm.keys()):
            if "-" not in k: continue
            a,b = k.split("-",1)
            na, nb = kp_len.get(a,0), kp_len.get(b,0)
            m = fm[k][()]
            if m.size == 0: continue
            m = m if m.shape[0]==2 else m.T
            keep = (m[0] >=0) & (m[0] < na) & (m[1] >=0) & (m[1] < nb)
            if not np.all(keep):
                dd = m.shape[1] - int(np.count_nonzero(keep))
                del fm[k]
                fm.create_dataset(k, data=m[:,keep].astype(np.int32,copy=False), compression="gzip", compression_opts=9)
                fixed += 1; dropped += dd
    print(f"[ok] sanitized {path.name}: fixed_pairs={fixed}, dropped_cols={dropped}")
for p in targets:
    sanitize(p)
PY

echo "[4] pack 生成 + multiview(run_0) を一度に実行（stereo無効 / 可視化OFF）"
rm -rf "${IMB}/jobs"/* || true
cd "${IMB}"
python3 "${IMB}/run.py" \
  --run_mode interactive \
  --json_method "${SUB}/config_jamma.json" \
  --dataset phototourism --subset val --scene "${SCENE}" \
  --scenes_phototourism_val "${SCENE}" \
  --eval_stereo False \
  --eval_multiview True \
  --num_runs_val_multiview 1 \
  --run_viz False --run_viz_debug False \
  --num_viz_stereo_pairs 0 --num_viz_stereo_pairs_debug 0 \
  --max_num_images_viz_multiview 0 \
  --path_data "${DATA}" \
  --path_results "${IMB}/results" \
  --path_pack "${IMB}/packed-val" \
  --num_opencv_threads 0 --opencv_seed 0

echo "[5] pack 内 pairs.txt を bag=5 のみ『隣接ペアだけ』に上書き"
python3 - <<'PY'
from pathlib import Path
PACK_ROOT = Path("/home/ach17765lb/image-matching-benchmark/packed-val/phototourism/reichstag")
def read_list(fp):
    xs=[]
    for line in fp.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith("#"): 
            continue
        xs.append(line)
    return xs
edited=0
for sub in sorted(PACK_ROOT.rglob("*")):
    img = sub/"images.txt"
    pr  = sub/"pairs.txt"
    if not (sub.is_dir() and img.exists() and pr.exists()):
        continue
    imgs = read_list(img)
    if len(imgs) != 5:
        continue  # bag=5 だけ対象
    pairs = [f"{imgs[i]} {imgs[i+1]}" for i in range(len(imgs)-1)]  # 隣接のみ（片方向）
    pr.write_text("\n".join(pairs) + ("\n" if pairs else ""))
    edited += 1
print(f"[ok] rewrote adjacent-only pairs for bag=5 subsets: {edited}")
PY

echo "[6] 既存の multiview(run_0) 出力を消して、隣接のみで再計算（pack再利用 / stereo無し / 可視化無し）"
# 既存 run_0（フルペアで計算された）の成果物を削除して強制再実行
rm -rf "${IMB}/results/phototourism/${SCENE}/jamma-kp_2048_jamma-desc/jamma-match/no_filter/multiview" || true
rm -rf "${IMB}/jobs"/* || true
python3 "${IMB}/run.py" \
  --run_mode interactive \
  --json_method "${SUB}/config_jamma.json" \
  --dataset phototourism --subset val --scene "${SCENE}" \
  --scenes_phototourism_val "${SCENE}" \
  --eval_stereo False \
  --eval_multiview True \
  --num_runs_val_multiview 1 \
  --skip_packing True \
  --run_viz False --run_viz_debug False \
  --num_viz_stereo_pairs 0 --num_viz_stereo_pairs_debug 0 \
  --max_num_images_viz_multiview 0 \
  --path_data "${DATA}" \
  --path_results "${IMB}/results" \
  --path_pack "${IMB}/packed-val" \
  --num_opencv_threads 0 --opencv_seed 0

echo "[7] 最終パッキング（pack_res.py 直叩き / stereo 無し / multiview 有り）"
rm -rf "${IMB}/jobs"/* || true
cd "${IMB}"
python3 "${IMB}/pack_res.py" \
  --json_method "${SUB}/config_jamma.json" \
  --dataset phototourism --subset val --scene "${SCENE}" \
  --scenes_phototourism_val "${SCENE}" \
  --path_results "${IMB}/results" \
  --path_pack    "${IMB}/packed-val" \
  --num_runs_val_multiview 1 \
  --eval_multiview True \
  --eval_stereo False

echo "[done] ${SCENE} / bag=5 / 隣接ペアのみ（multiview）完了。"
