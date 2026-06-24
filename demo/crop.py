import cv2, json, math, numpy as np , zipfile
from pathlib import Path
from numpy.random import default_rng
import random

SRC = "/home/ach17765lb/JamMa/demo/PicassoGuernica2.jpg"
OUTDIR = Path("/home/ach17765lb/JamMa/demo/guernica_seq_fixed"); OUTDIR.mkdir(parents=True, exist_ok=True)
BASE_XY = (300, 300)          # img0 の切り出し左上座標（元画像座標）
CROP = 832
DX_RANGE = (15, 30)         # x 平行移動の希望量（img0 から）
DY_RANGE = (1, 5)          # y 平行移動の希望量（img0 から）

ROT_RANGE = (-10, 10)         # 回転角（deg）※0は再サンプリング
SHX_RANGE = (-10, 10)         # シアーx（deg）※0は再サンプリング
SHY_RANGE = (-10, 10)         # シアーy（deg）※0は再サンプリング

SEED = 12                  # 再現用
rng = default_rng(SEED)
random.seed(SEED)

# ---------- 幾何ユーティリティ ----------
def H_translate(dx, dy):
    return np.array([[1,0,dx],[0,1,dy],[0,0,1.0]], float)

def H_rotate_about(cx, cy, deg):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1.0]], float)
    return H_translate(cx,cy) @ R @ H_translate(-cx,-cy)

def H_shear_about(cx, cy, shx_deg, shy_deg):
    shx = math.tan(math.radians(shx_deg))
    shy = math.tan(math.radians(shy_deg))
    S = np.array([[1, shx, 0],[shy, 1, 0],[0,0,1.0]], float)
    return H_translate(cx,cy) @ S @ H_translate(-cx,-cy)

def non_zero_int(lo, hi):
    while True:
        v = random.randint(lo, hi)
        if v != 0: return v

# ---------- 元画像 ----------
src = cv2.imread(SRC, cv2.IMREAD_COLOR)
assert src is not None, f"Failed to read: {SRC}"
H0, W0 = src.shape[:2]
print(f"Source image size: W={W0}, H={H0}")
cx, cy = (W0-1)/2.0, (H0-1)/2.0

# ---------- 出力用 ----------
meta = {
    "source_path": SRC,
    "crop_size": CROP,
    "base_top_left_xy": list(BASE_XY),
    "frames": []
}

x0, y0 = BASE_XY
img0 = src[y0:y0+CROP, x0:x0+CROP].copy()
cv2.imwrite(str(OUTDIR/"img0.png"), img0)
np.save(OUTDIR/"H_00.npy", np.eye(3))
absH = [np.eye(3)]

prev_x, prev_y = x0, y0

# ---------- img1〜img9 ----------
for k in range(1, 10):
    dx = random.randint(*DX_RANGE)
    dy = random.randint(*DY_RANGE)
    prev_x += dx
    prev_y += dy
    px = min(prev_x, W0 - CROP)
    py = min(prev_y, H0 - CROP)
    print(f"img{k}: crop top-left=({px},{py}), delta=({dx},{dy})")

    rot = non_zero_int(*ROT_RANGE)
    shx = non_zero_int(*SHX_RANGE)
    shy = non_zero_int(*SHY_RANGE)

    Hglob = H_shear_about(cx, cy, shx, shy) @ H_rotate_about(cx, cy, rot)

    # 有効領域マスクを作成
    valid_mask = cv2.warpPerspective(
        np.ones(src.shape[:2], np.uint8) * 255,
        Hglob, (W0, H0),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # 通常ワープ（黒埋め）
    warped_black = cv2.warpPerspective(
        src, Hglob, (W0, H0),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 同じ変換をかけた元画像を生成（補完用）
    warped_src = cv2.warpPerspective(
        src, Hglob, (W0, H0),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 有効画素以外を元画像で埋める（反転ナシ）
    invalid = (valid_mask == 0)
    warped = warped_black.copy()
    warped[invalid] = warped_src[invalid]

    # クロップ
    imgk = warped[py:py+CROP, px:px+CROP].copy()
    cv2.imwrite(str(OUTDIR/f"img{k}.png"), imgk)

    # ホモグラフィ（img0→imgk）
    H_0k = H_translate(-px, -py) @ Hglob @ H_translate(x0, y0)
    np.save(OUTDIR/f"H_0{k}.npy", H_0k)
    absH.append(H_0k)

    # 隣接ホモグラフィ
    if k >= 2:
        H_prevk = absH[k] @ np.linalg.inv(absH[k-1])
        np.save(OUTDIR/f"H_{k-1}{k}.npy", H_prevk)

    meta["frames"].append({
        "index": k,
        "rotation_deg": rot,
        "shear_x_deg": shx,
        "shear_y_deg": shy,
        "crop_top_left_xy": [int(px), int(py)],
        "delta_from_prev": [dx, dy]
    })

# ---------- 保存 ----------
with open(OUTDIR/"meta.json", "w") as f:
    json.dump(meta, f, indent=2)
