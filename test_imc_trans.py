#!/usr/bin/env python3
"""
Convert JamMa npy outputs to IMC2021 HDF5 format AND emit a config JSON.

Outputs per scene:
  <out_root>/phototourism/<scene>/
    keypoints.h5
    matches_stereo_0.h5      # keys: "imgA-imgB", shape: 2xN (int32)
    matches_multiview.h5      # keys: "imgA-imgB", shape: 2xN (int32)
    descriptors.h5            # zeros (required even with custom matches)

Also writes method config:
  <out_root>/config_jamma.json

Input layouts supported (per scene):
  A) <jamma_root>/<scene>/jamma/*.npy
  B) <jamma_root>/phototourism/<scene>/jamma/*.npy
"""
import argparse, json
from pathlib import Path
import numpy as np
import h5py


def add(h5: h5py.File, name: str, arr, dtype=None):
    if name in h5:
        del h5[name]
    a = np.asarray(arr if dtype is None else arr.astype(dtype))
    h5.create_dataset(name, data=a, compression="gzip", compression_opts=9)


def to_flat_key(a: str, b: str) -> str:
    # IMC は "imgA-imgB" のフラットキーを期待
    return f"{a}-{b}"


def matches_to_2xN(m: np.ndarray) -> np.ndarray:
    """Ensure matches are shape 2xN (int32). Accepts Nx2 or 2xN."""
    if m.ndim != 2:
        raise ValueError(f"unexpected matches ndim: {m.ndim}")
    if m.size == 0:
        return np.empty((2, 0), dtype=np.int32)
    if m.shape[1] == 2:
        m = m.T
    elif m.shape[0] == 2:
        pass
    else:
        raise ValueError(f"unexpected matches shape: {m.shape}")
    return m.astype(np.int32)


def find_scene_dirs(jamma_root: Path):
    """Return a list of scene dirs that each contains a 'jamma' subdir."""
    dirs = []
    # Option B: jamma_root/phototourism/<scene>/jamma
    pt_root = jamma_root / "phototourism"
    if pt_root.exists():
        for s in sorted(p for p in pt_root.iterdir() if p.is_dir()):
            if (s / "jamma").exists():
                dirs.append(s)
    # Option A: jamma_root/<scene>/jamma
    for s in sorted(p for p in jamma_root.iterdir() if p.is_dir()):
        if (s / "jamma").exists() and s not in dirs:
            dirs.append(s)
    return dirs


def fill_missing_pairs(kp_h5_path: Path, match_h5_path: Path):
    """全画像の全組み合わせについて、欠けているキーに 2x0 の空マッチを作成（A-B と B-A の両方）。"""
    with h5py.File(kp_h5_path, "r") as fkp:
        names = sorted(list(fkp.keys()))
    need_keys = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            need_keys.append(f"{a}-{b}")
            need_keys.append(f"{b}-{a}")
    created = 0
    with h5py.File(match_h5_path, "a") as fm:
        for k in need_keys:
            if k not in fm:
                fm.create_dataset(k, data=np.empty((2, 0), dtype=np.int32),
                                  compression="gzip", compression_opts=9)
                created += 1
    print(f"[ok] filled {created} empty pairs in {match_h5_path.name}")


def convert_one_scene(in_dir: Path, out_dir: Path, num_keypoints: int, desc_dim: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    kp_h5   = h5py.File(out_dir / "keypoints.h5", "w")
    mst0_h5 = h5py.File(out_dir / "matches_stereo_0.h5", "w")
    mmv_h5  = h5py.File(out_dir / "matches_multiview.h5", "w")
    desc_h5 = h5py.File(out_dir / "descriptors.h5", "w")

    # keypoints + dummy descriptors
    kp_files = sorted(in_dir.glob("*.keypoints.npy"))
    stems = [p.stem.replace(".keypoints", "") for p in kp_files]
    for stem in stems:
        kpf = in_dir / f"{stem}.keypoints.npy"
        kp = np.load(kpf)
        kp = kp[:num_keypoints].astype(np.float32)
        add(kp_h5, stem, kp)
        add(desc_h5, stem, np.zeros((len(kp), desc_dim), dtype=np.float32))

    # matches (write as flat keys "A-B", shape 2xN)
    for mfile in sorted(in_dir.glob("*.matches.npy")):
        pair = mfile.stem.replace(".matches", "")
        if "__" not in pair:
            print(f"[skip] invalid pair name: {pair}")
            continue
        name_i, name_j = pair.split("__", 1)
        M = np.load(mfile)
        M = matches_to_2xN(M)
        k = to_flat_key(name_i, name_j)
        add(mst0_h5, k, M)
        add(mmv_h5,  k, M)

    kp_h5.close(); mst0_h5.close(); mmv_h5.close(); desc_h5.close()

    # 欠損ペアを空で補完
    fill_missing_pairs(out_dir / "keypoints.h5", out_dir / "matches_stereo_0.h5")
    fill_missing_pairs(out_dir / "keypoints.h5", out_dir / "matches_multiview.h5")


def write_config(out_root: Path, json_label: str, kp: str, desc: str, match: str, num_keypoints: int):
    # スキーマ準拠：config_common に 'match' は置かない（各タスクで custom_matches_name を指定）
    cfg = [
        {
            "metadata": {
                "publish_anonymously": False,
                "authors": "JamMa team",
                "contact_email": "you@example.com",
                "method_name": "JamMa_5bag_adjacent",
                "method_description": "JamMa; adjacent pairs from 5bag_*; Phototourism val."
            },
            "config_common": {
                "json_label": json_label,
                "keypoint": kp,
                "descriptor": desc,
                "num_keypoints": int(num_keypoints)
            },
            "config_phototourism_stereo": {
                "use_custom_matches": True,
                "custom_matches_name": match,
                "geom": { "method": "cv2-8pt" }
            },
            "config_phototourism_multiview": {
                "use_custom_matches": True,
                "custom_matches_name": match,
                "colmap": {}
            }
        }
    ]
    out_path = out_root / "config_jamma.json"
    out_path.write_text(json.dumps(cfg, indent=2))
    print(f"[ok] wrote config -> {out_path}")


def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--jamma_root", default="/home/ach17765lb/JamMa/outputs/",
                    help="Path to npy outputs (per scene folders).")
    ap.add_argument("--out_root",   default="/home/ach17765lb/JamMa/imc_submission",
                    help="Output root for HDF5 files.")
    ap.add_argument("--num_keypoints", type=int, default=2048, help="2048 or 8000")
    ap.add_argument("--desc_dim", type=int, default=256, help="descriptor dimension for zeros")
    ap.add_argument("--json_label", default="jamma_5bag_adj")
    ap.add_argument("--kp_name",    default="jamma-kp")     # ハイフン表記で統一
    ap.add_argument("--desc_name",  default="jamma-desc")
    ap.add_argument("--match_name", default="jamma-match")
    ap.add_argument("--scenes", nargs="*", default=None, help="evaluate only these scenes (e.g., reichstag)")
    args = ap.parse_args()

    jr = Path(args.jamma_root)
    oroot = Path(args.out_root)
    scenes_dirs = find_scene_dirs(jr)

    if args.scenes:
        names = set(args.scenes)
        scenes_dirs = [s for s in scenes_dirs if s.name in names]

    print(f"[info] jamma_root = {jr}")
    if not scenes_dirs:
        print(f"[err] no scenes with 'jamma' found under {jr}")
        return
    print(f"[info] found scenes: {[s.name for s in scenes_dirs]}")

    # always write under out_root/phototourism/<scene>
    ds_root = oroot / "phototourism"
    ds_root.mkdir(parents=True, exist_ok=True)

    for sdir in scenes_dirs:
        scene = sdir.name
        in_dir = sdir / "jamma"
        out_dir = ds_root / scene
        print(f"[scene] {scene}: {in_dir} -> {out_dir}")
        convert_one_scene(in_dir, out_dir, args.num_keypoints, args.desc_dim)
        print(f"[ok] wrote HDF5 for {scene}")

    write_config(oroot, args.json_label, args.kp_name, args.desc_name, args.match_name, args.num_keypoints)
    print(f"[done] converted to HDF5 at {oroot}")


if __name__ == "__main__":
    main()

