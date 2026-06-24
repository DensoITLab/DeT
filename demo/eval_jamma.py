import os
import json
import numpy as np
import cv2
from numpy.linalg import norm
import random
import glob
import re

# ========= 設定 =========
IMAGES_DIR  = "/home/ach17765lb/JamMa/demo/guernica_seq_fixed" # 画像ディレクトリパス
PAIR_H_DIR  = "/home/ach17765lb/JamMa/demo/guernica_seq_fixed" # 相対ホモグラフィディレクトリパス (H_01.npy, H_12.npy など)
ABS_H_DIR   = "/home/ach17765lb/JamMa/demo/guernica_seq_fixed" # 絶対ホモグラフィディレクトリパス (H_01.npy, H_02.npy など)

IMAGES_DIR  = "/home/ach17765lb/JamMa/assets/carla1"
PAIR_H_DIR  = "/home/ach17765lb/JamMa/assets/carla1"
ABS_H_DIR   = "/home/ach17765lb/JamMa/assets/carla1"

IMAGES_DIR  = "/home/ach17765lb/JamMa/assets/carla2"
PAIR_H_DIR  = "/home/ach17765lb/JamMa/assets/carla2"
ABS_H_DIR   = "/home/ach17765lb/JamMa/assets/carla2"

IMAGES_DIR  = "/home/ach17765lb/JamMa/assets/carla3"
PAIR_H_DIR  = "/home/ach17765lb/JamMa/assets/carla3"
ABS_H_DIR   = "/home/ach17765lb/JamMa/assets/carla3"

IMAGES_DIR  = "/home/ach17765lb/JamMa/assets/carla4"
PAIR_H_DIR  = "/home/ach17765lb/JamMa/assets/carla4"
ABS_H_DIR   = "/home/ach17765lb/JamMa/assets/carla4"

IMAGES_DIR  = "/home/ach17765lb/JamMa/assets/carla5"
PAIR_H_DIR  = "/home/ach17765lb/JamMa/assets/carla5"
ABS_H_DIR   = "/home/ach17765lb/JamMa/assets/carla5"

IMAGES_DIR  = "/home/ach17765lb/JamMa/assets/test_subset10"
PAIR_H_DIR  = "/home/ach17765lb/JamMa/assets/test_subset10"
ABS_H_DIR   = "/home/ach17765lb/JamMa/assets/test_subset10"


TRACKS_JSON = "/home/ach17765lb/JamMa/demo/output_carla5/track_result.json" # トラック結果JSONパス
OUT_PATH    = "/home/ach17765lb/JamMa/demo/overlay_concat_carla5_all.png"   # 出力画像パス

TRACKS_JSON = "/home/ach17765lb/JamMa/demo/output_guernica_seq_fixed/track_result.json"
OUT_PATH    = "/home/ach17765lb/JamMa/demo/overlay_concat_guernica_seq_fixed.png"

TRACKS_JSON = "/home/ach17765lb/data/phototourism/tracks_epipolar_jamma.json"
OUT_PATH    = "/home/ach17765lb/JamMa/demo/overlay_concat_phototourism.png"

TRACKS_JSON = "/home/ach17765lb/JamMa/demo/output_test_subset10/track_result.json"
OUT_PATH    = "/home/ach17765lb/JamMa/demo/overlay_concat_test_subset10.png"

THRESH_PX   = 3.0   # 成功とみなす誤差閾値（単位ピクセル）
GRID_SPACING = 8    # グリッド間隔（ピクセル）
GAP = 10            # フレーム間の隙間ピクセル数

# 色,太さ等設定
GT_POINT_COLOR     = (0, 255, 0)
GT_LINE_COLOR      = (0, 200, 0)
PRED_POINT_COLOR   = (255, 0, 255)
PRED_LINE_COLOR    = (200, 0, 200)
GRID_COLOR         = (180, 180, 180)
GRID_ALPHA         = 0.2
RADIUS, THICKNESS  = 2, 1
THICKNESS_LINE     = 1
THICKNESS_POINT    = 2
LINE_ALPHA         = 0.3

# 描画フラグ
DRAW_GT_LINE      = True   # GT線を描画するか
DRAW_PRED_LINE    = True   # 予測線を描画するか
DRAW_TRACK_ID     = False  # 各点に track_id を描画するか
DRAW_ALL_TRACKS   = True   # 全トラックを描画するか（Falseなら成功トラックのみ）
NUM_VALID_PAIR    = 9      # 最低ペア数（これ未満のトラックは無視）(9なら 9+1で画像10枚でトラックがつながっているものだけ対象)

DRAW_GT_IF_AVAILABLE = True  # GTが存在しないときは自動スキップ

START_ID = 0  # start_id がこれと等しいトラックのみ評価・描画対象とする

# ランダム描画制御
SEED = 42
DRAW_LIMIT       = 5000  # 描画するトラック数 (GTあり)
DRAW_LIMIT_NO_GT = 300   # GTなしのときの描画上限

# =========================

def load_frames(images_dir: str):
    """frame_*.jpg / img*.png の連番を自動検出して昇順に読み込む"""
    paths = sorted(
        glob.glob(os.path.join(images_dir, "frame_*.jpg")),
        key=lambda p: int(re.search(r'(\d+)(?=\.jpg$)', os.path.basename(p)).group(1))
    )
    if not paths:
        paths = sorted(
            glob.glob(os.path.join(images_dir, "img*.jpg")),
            key=lambda p: int(re.search(r'(\d+)(?=\.jpg$)', os.path.basename(p)).group(1))
        )
    if not paths:
        raise FileNotFoundError(f"frame_*.jpg / img*.png が見つかりません: {images_dir}")
    frames = [cv2.imread(p) for p in paths]
    missing = [i for i, im in enumerate(frames) if im is None]
    if missing:
        raise FileNotFoundError(f"読めなかったフレーム: {missing} → {paths[missing[0]]}")
    return frames, paths


def load_pairwise_H(i: int, j: int, base_dir: str = PAIR_H_DIR):
    for name in [f"H_{i}{j}.npy", f"H_{i:02d}{j:02d}.npy", f"H_{i}_{j}.npy"]:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return np.load(p)
    return None


def load_abs_H(k: int, base_dir: str = ABS_H_DIR):
    for name in [f"H_{k:02d}.npy", f"H_{k}.npy"]:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return np.load(p)
    return None


def get_H_between(i: int, j: int):
    if i == j:
        return np.eye(3, dtype=float)
    if j < i:
        H_inv = get_H_between(j, i)
        return np.linalg.inv(H_inv) if H_inv is not None else None

    H = load_pairwise_H(i, j)
    if H is not None:
        return H

    H_acc = np.eye(3, dtype=float)
    for k in range(i, j):
        H_k1 = load_pairwise_H(k, k + 1)
        if H_k1 is None:
            H_i, H_j = load_abs_H(i), load_abs_H(j)
            if H_i is None or H_j is None:
                # --- GTファイルが存在しない場合 ---
                if DRAW_GT_IF_AVAILABLE:
                    print(f"⚠️ GT未検出: H_{k}{k+1}.npy が存在しません。GTをスキップします。")
                    return None
                else:
                    raise FileNotFoundError(f"H_{k}{k+1}.npy が見つかりません")
            return H_j @ np.linalg.inv(H_i)
        H_acc = H_k1 @ H_acc
    return H_acc


def apply_H_point(pxy, H):
    v = np.array([pxy[0], pxy[1], 1.0], dtype=float)
    w = H @ v
    return np.array([w[0] / w[2], w[1] / w[2]], dtype=float)


def build_pred_sequence(points, start_id):
    L = len(points)
    pred = np.zeros_like(points, dtype=float)
    pred[0] = points[0]
    for k in range(L - 1):
        H = get_H_between(start_id + k, start_id + k + 1)
        if H is None:
            # GTがない場合は以降の予測をコピー（変化なし）
            pred[k + 1:] = pred[k]
            print(f"🚫 GTなし: frame {start_id+k}→{start_id+k+1} をスキップ")
            break
        pred[k + 1] = apply_H_point(pred[k], H)
    return pred


def draw_grid_soft(img, spacing=8, color=(180, 180, 180), alpha=0.2):
    grid = img.copy()
    h, w = img.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(grid, (x, 0), (x, h), color, 1)
    for y in range(0, h, spacing):
        cv2.line(grid, (0, y), (w, y), color, 1)
    return cv2.addWeighted(grid, alpha, img, 1 - alpha, 0)


# ========= 可変解像度対応のキャンバス生成 =========

def build_concat_canvas(frames, gap=GAP):
    """
    フレームごとのサイズがバラバラでもきれいに横連結するキャンバスを作る。
    返り値:
      canvas: 連結済み背景画像
      x_offsets: 各フレームの左上Xオフセット
      y_offsets: 各フレームの左上Yオフセット（縦方向は中央寄せ）
    """
    N = len(frames)
    heights = [f.shape[0] for f in frames]
    widths  = [f.shape[1] for f in frames]

    canvas_h = max(heights)
    canvas_w = sum(widths) + gap * (N - 1)

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    x_offsets = []
    y_offsets = []
    offset_x = 0
    for i in range(N):
        h, w = heights[i], widths[i]
        # 縦方向は中央寄せ（上揃えにしたければ y0=0）
        y0 = (canvas_h - h) // 2
        frame = draw_grid_soft(frames[i].copy(), GRID_SPACING, GRID_COLOR, GRID_ALPHA)
        canvas[y0:y0 + h, offset_x:offset_x + w] = frame
        x_offsets.append(offset_x)
        y_offsets.append(y0)
        offset_x += w + gap

    return canvas, x_offsets, y_offsets


# ==================== 評価関数 ====================

def evaluate_one_track(rec, thresh_px=THRESH_PX, img_sizes=None):
    """
    GTが画像範囲外になる点も検出
    img_sizes: [(h0, w0), (h1, w1), ...] というフレームごとのサイズリスト
    """
    pts = np.array(rec["points"], dtype=float)
    L = len(pts)
    start_id = int(rec["start_id"])
    frame_ids = np.arange(start_id, start_id + L)
    pred = build_pred_sequence(pts, start_id)

    details = []
    ok_cnt = 0
    out_cnt = 0
    errs = []
    for k in range(L - 1):
        gt = pred[k + 1]
        e = float(norm(gt - pts[k + 1]))
        ok = e <= thresh_px

        # 対応するターゲットフレーム k+1 のサイズで in_bounds 判定
        if img_sizes is not None and frame_ids[k + 1] < len(img_sizes):
            h_t, w_t = img_sizes[frame_ids[k + 1]]
        else:
            h_t, w_t = 1e9, 1e9  # サイズ情報がなければほぼ無限大扱い

        in_bounds = (0 <= gt[0] < w_t) and (0 <= gt[1] < h_t)
        if not in_bounds:
            out_cnt += 1
        errs.append(e)
        ok_cnt += int(ok)
        details.append({
            "frame_pair": (int(frame_ids[k]), int(frame_ids[k + 1])),
            "pred": gt.tolist(),
            "gt": pts[k + 1].tolist(),
            "error_px": e,
            "success": ok,
            "in_bounds": in_bounds
        })
    return {
        "pred": pred,
        "pts": pts,
        "frame_ids": frame_ids,
        "num_pairs": max(0, L - 1),
        "num_success": ok_cnt,
        "errors": errs,
        "details": details,
        "max_error": max(errs) if errs else 0.0,
        "ng_count": (len(errs) - ok_cnt) if errs else 0,
        "out_of_bounds": out_cnt
    }


def draw_tracks_with_eval(track_ids, eval_map, frames, out_path):
    N = len(frames)
    canvas, x_offsets, y_offsets = build_concat_canvas(frames, GAP)

    total_out = 0

    for tid in track_ids:
        ev = eval_map[tid]
        pts, pred, fids = ev["pts"], ev["pred"], ev["frame_ids"]

        gt_visible = True
        for k, d in enumerate(ev["details"]):
            f0, f1 = int(fids[k]), int(fids[k + 1])

            if f0 >= N or f1 >= N:
                continue

            off0_x = x_offsets[f0]
            off1_x = x_offsets[f1]
            off0_y = y_offsets[f0]
            off1_y = y_offsets[f1]

            p0_pred = (int(pts[k][0]     + off0_x), int(pts[k][1]     + off0_y))
            p1_pred = (int(pts[k + 1][0] + off1_x), int(pts[k + 1][1] + off1_y))
            q0_gt   = (int(pred[k][0]    + off0_x), int(pred[k][1]    + off0_y))
            q1_gt   = (int(pred[k + 1][0]+ off1_x), int(pred[k + 1][1]+ off1_y))

            # GT が画面外になったら以降は非表示
            if gt_visible and not d["in_bounds"]:
                gt_visible = False
                total_out += 1

            # GT描画（in_bounds の間のみ）
            if gt_visible and DRAW_GT_LINE:
                overlay = canvas.copy()
                cv2.line(overlay, q0_gt, q1_gt, GT_LINE_COLOR, THICKNESS_LINE, cv2.LINE_AA)
                canvas = cv2.addWeighted(overlay, LINE_ALPHA, canvas, 1 - LINE_ALPHA, 0)
                cv2.circle(canvas, q0_gt, RADIUS, GT_POINT_COLOR, -1, cv2.LINE_AA)
                cv2.circle(canvas, q1_gt, RADIUS, GT_POINT_COLOR, -1, cv2.LINE_AA)

            # 予測結果は常に描画
            if DRAW_PRED_LINE:
                overlay = canvas.copy()
                cv2.line(overlay, p0_pred, p1_pred, PRED_LINE_COLOR, THICKNESS_LINE, cv2.LINE_AA)
                canvas = cv2.addWeighted(overlay, LINE_ALPHA, canvas, 1 - LINE_ALPHA, 0)
                cv2.circle(canvas, p0_pred, RADIUS, PRED_POINT_COLOR, -1, cv2.LINE_AA)
                cv2.circle(canvas, p1_pred, RADIUS, PRED_POINT_COLOR, -1, cv2.LINE_AA)

            if DRAW_TRACK_ID:
                for pt in [q0_gt, q1_gt, p0_pred, p1_pred]:
                    pos = (pt[0] - 6, pt[1] - 6)
                    cv2.putText(canvas, str(tid), pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                                (255, 255, 255), 1, cv2.LINE_AA)

    print(f"\n⚠️ GTが画面外になり以降非表示となったトラック: {total_out} 箇所")
    cv2.imwrite(out_path, canvas)
    print(f"\n✅ 連結画像を保存しました: {out_path}")


# ========================== GTなしモード ==========================

def evaluate_one_track_no_gt(rec):
    """GTなし: 誤差評価を行わず予測線のみ描画"""
    pts = np.array(rec["points"], dtype=float)
    L = len(pts)
    start_id = int(rec["start_id"])
    frame_ids = np.arange(start_id, start_id + L)

    return {
        "pts": pts,
        "frame_ids": frame_ids,
        "num_pairs": max(0, L - 1),
        "pred": None,   # GTがないためNone
        "details": [],
        "ng_count": 0,
        "out_of_bounds": 0,
    }


def draw_tracks_pred_only(track_ids, eval_map, frames, out_path):
    """GTなしの場合の純予測線描画（全点にtrack_id表示、開始=緑／中間=白／終了=赤）"""
    N = len(frames)
    canvas, x_offsets, y_offsets = build_concat_canvas(frames, GAP)

    for tid in track_ids:
        ev = eval_map[tid]
        pts, fids = ev["pts"], ev["frame_ids"]
        num_pairs = ev["num_pairs"]

        # --- 線と点を描画 ---
        for k in range(num_pairs):
            f0, f1 = int(fids[k]), int(fids[k + 1])
            if f0 >= N or f1 >= N:
                continue
            off0_x, off1_x = x_offsets[f0], x_offsets[f1]
            off0_y, off1_y = y_offsets[f0], y_offsets[f1]

            p0 = (int(pts[k][0]     + off0_x), int(pts[k][1]     + off0_y))
            p1 = (int(pts[k + 1][0] + off1_x), int(pts[k + 1][1] + off1_y))

            overlay = canvas.copy()
            cv2.line(overlay, p0, p1, PRED_LINE_COLOR, THICKNESS_LINE, cv2.LINE_AA)
            canvas = cv2.addWeighted(overlay, LINE_ALPHA, canvas, 1 - LINE_ALPHA, 0)

            cv2.circle(canvas, p0, RADIUS, PRED_POINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, p1, RADIUS, PRED_POINT_COLOR, -1, cv2.LINE_AA)

        # --- 各点に track_id を描画（開始=緑／中間=白／終了=赤） ---
        if DRAW_TRACK_ID:
            for i, (pt, fid) in enumerate(zip(pts, fids)):
                if fid >= N:
                    continue
                off_x = x_offsets[fid]
                off_y = y_offsets[fid]
                px, py = int(pt[0] + off_x), int(pt[1] + off_y)

                offset_x = random.randint(-3, 3)
                offset_y = random.randint(-3, 3)

                if i == 0:
                    text_color = (0, 255, 0)   # 開始: 緑
                elif i == len(pts) - 1:
                    text_color = (0, 0, 255)   # 終了: 赤
                else:
                    text_color = (255, 255, 255)  # 中間: 白

                cv2.putText(
                    canvas,
                    str(tid),
                    (px + 2 + offset_x, py - 2 + offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

    cv2.imwrite(out_path, canvas)
    print(f"\n✅ GTなしモードで描画完了: {out_path}")


# ========================== diff_points 可視化 ==========================

def visualize_all_diff_points_horizontal(frames, tracks, draw_ids, out_path, gap=10):
    """
    draw_ids に含まれる全トラックの diff_points を、
    フレームサイズがバラバラでも frame1~N 横連結で1枚に描画する。
    """
    tracks_with_diff = [tracks[tid] for tid in draw_ids if "diff_points" in tracks[tid]]
    if not tracks_with_diff:
        print("⚠️ diff_points を持つトラックがありません。")
        return

    num_pairs = len(tracks_with_diff[0]["diff_points"])
    N = num_pairs + 1

    # 元コードは frame1〜N を使っていたのでそれに合わせる
    if len(frames) < N + 1:
        print("⚠️ フレーム数が diff_points のフレーム数に足りていません。")
        return

    use_frames = frames[1:N + 1]  # frame_1〜frame_N
    canvas, x_offsets, y_offsets = build_concat_canvas(use_frames, gap)

    for tid, rec in zip(draw_ids, tracks_with_diff):
        diff_pts_all = np.array(rec["diff_points"], dtype=object)

        START_COLOR = (0, 255, 0)  # 始点: 緑
        END_COLOR   = (0, 0, 255)  # 終点: 赤

        for i, pair in enumerate(diff_pts_all):
            if len(pair) != 2:
                continue
            if i >= len(x_offsets) - 1:
                continue

            # frame_(i+1) と frame_(i+2) に対応
            p0 = (int(pair[0][0] + x_offsets[i]),     int(pair[0][1] + y_offsets[i]))
            p1 = (int(pair[1][0] + x_offsets[i + 1]), int(pair[1][1] + y_offsets[i + 1]))

            cv2.circle(canvas, p0, 2, START_COLOR, -1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, END_COLOR,   -1, cv2.LINE_AA)

            cv2.putText(canvas, str(tid), (p0[0] + 3, p0[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    print(f"✅ 全トラックの diff_points を横連結形式で1枚に描画しました: {out_path}")


# ========================== main ==========================

def main():
    with open(TRACKS_JSON, "r") as f:
        tracks = json.load(f)
        #tracks = tracks['per_bag']
        #tracks = tracks[0]['tracks']

    # ✅ start_id == START_ID のものだけ抽出
    tracks = {tid: rec for tid, rec in tracks.items()
              if int(rec.get("start_id", -1)) == START_ID}
    print(f"🎯 start_id == {START_ID} のトラックのみ選択: {len(tracks)} 本")

    if not tracks:
        print(f"⚠️ start_id == {START_ID} のトラックがありません。終了します。")
        return

    frames, frame_paths = load_frames(IMAGES_DIR)
    frame_sizes = [(im.shape[0], im.shape[1]) for im in frames]

    gt_exists = any(
        os.path.exists(os.path.join(PAIR_H_DIR, f))
        for f in os.listdir(PAIR_H_DIR)
        if f.endswith(".npy")
    )
    if not gt_exists:
        print("⚠️ GTファイル未検出: 誤差評価をスキップし、予測線のみ描画します。")
        eval_map = {tid: evaluate_one_track_no_gt(rec) for tid, rec in tracks.items()}

        # ✅ トラック数制御（GTなし）
        all_ids = list(eval_map.keys())
        random.seed(SEED)
        if DRAW_LIMIT_NO_GT < len(all_ids):
            all_ids = [tid for tid, ev in eval_map.items()
                       if ev["num_pairs"] >= NUM_VALID_PAIR]
            if DRAW_LIMIT_NO_GT < len(all_ids):
                draw_ids = random.sample(all_ids, DRAW_LIMIT_NO_GT)
                print(f"🎯 GTなしモード: {DRAW_LIMIT_NO_GT} 本をランダム選択(総数{len(all_ids)})")
            else:
                draw_ids = all_ids
                print(f"🎯 GTなしモード: {len(draw_ids)} 本全て描画（総数 {len(all_ids)}）")
        else:
            draw_ids = all_ids
            print(f"🎯 GTなしモード: {len(draw_ids)} 本全て描画（総数 {len(all_ids)}）")

        draw_tracks_pred_only(draw_ids, eval_map, frames, OUT_PATH)

        # diff_points 可視化するときはコメント外して使う
        # combined_diff_out = OUT_PATH.replace(".png", "_diff_all_horizontal.png")
        # visualize_all_diff_points_horizontal(frames, tracks, draw_ids, combined_diff_out)
        return

    # ✅ GTあり: フレームごとのサイズを渡して評価
    eval_map = {tid: evaluate_one_track(rec, img_sizes=frame_sizes)
                for tid, rec in tracks.items()}

    if DRAW_ALL_TRACKS:
        all_ids = [tid for tid, ev in eval_map.items()
                   if ev["num_pairs"] >= NUM_VALID_PAIR]
        print(f"\n✅ 全トラック描画モード（合計 {len(all_ids)} 本, start_id=={START_ID} 限定）")
    else:
        all_ids = [tid for tid, ev in eval_map.items()
                   if ev["num_pairs"] >= NUM_VALID_PAIR and ev["ng_count"] == 0]
        print(f"\n✅ 全成功トラック（{NUM_VALID_PAIR}ペア以上, start_id=={START_ID} 限定）: {len(all_ids)}本")

    if not all_ids:
        print(f"⚠️ start_id=={START_ID} の該当トラックがありません。")
        return

    random.seed(SEED)
    if DRAW_LIMIT < len(all_ids):
        draw_ids = random.sample(all_ids, DRAW_LIMIT)
        print(f"🎯 {DRAW_LIMIT} 本をランダム選択: {draw_ids[:10]} ...")
    else:
        draw_ids = all_ids
        print(f"🎯 {len(draw_ids)} 本全て描画（総数 {len(all_ids)}）")

    print(f"検出フレーム数: {len(frames)} 例: {os.path.basename(frame_paths[0])} ... {os.path.basename(frame_paths[-1])}")
    draw_tracks_with_eval(draw_ids, eval_map, frames, OUT_PATH)


if __name__ == "__main__":
    main()
