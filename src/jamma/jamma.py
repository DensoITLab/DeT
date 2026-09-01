import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops.einops import rearrange

from src.jamma.utils.utils import (
    KeypointEncoder_wo_score,
    up_conv4,
    MLPMixerEncoderLayer,
    normalize_keypoints,
)
from src.jamma.mamba_module import JointMamba
from src.jamma.matching_module import CoarseMatching, FineSubMatching
from src.utils.profiler import PassThroughProfiler

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
INF = 1e9


_BASE_GRID_CACHE = {}

@torch.compile
def _get_base_grid(device, dtype):
    key = (device, dtype)
    if key not in _BASE_GRID_CACHE:
        xs = torch.arange(1, 6, device=device, dtype=dtype)
        ys = torch.arange(1, 6, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((xx / 3 - 1, yy / 3 - 1), dim=-1)  # [5,5,2]
        _BASE_GRID_CACHE[key] = base
    return _BASE_GRID_CACHE[key]

@torch.compile
def _project_7x7_to_5x5(inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Project 7x7 local feature patches into 5x5 patches with an offset.

    Args:
        inputs: [B, 49, C] feature patches (7x7 flattened).
        offsets: [B, 2] sub-pixel offsets (x, y) in pixel space.

    Returns:
        [B, 25, C] feature patches (5x5 flattened).
    """
    B, HW, C = inputs.shape
    assert HW == 49, f"Expected 49 (=7x7), got {HW}"
    assert offsets.shape == (B, 2), f"offsets should be (B, 2), got {offsets.shape}"

    # [B, 49, C] -> [B, C, 7, 7]
    x = inputs.permute(0, 2, 1).reshape(B, C, 7, 7)
    device = x.device

    # base 5x5 grid in normalized coordinates (-1..1).

    # convert pixel offsets to normalized coordinate offsets and add to base grid

    base = _get_base_grid(device=device, dtype=x.dtype)  # [5,5,2]
    grid = base.unsqueeze(0) + (offsets.view(B, 1, 1, 2) / 3)


    grid = grid.to(x.dtype)

    # sample features using bilinear sampling
    y = F.grid_sample(x, grid, align_corners=True, padding_mode="zeros")  # [B, C, 5, 5]

    # [B, C, 5, 5] -> [B, 25, C]
    y = y.flatten(2).permute(0, 2, 1).contiguous()
    return y


def _flat_idx_from_xy(
    xy: torch.Tensor,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    width: int = 416,
    height: int = 416,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert (x, y) coordinates to flattened indices with boundary filtering.

    Args:
        xy: (..., 2) coordinates [x, y].
        x_min, y_min, x_max, y_max: valid coordinate bounds (inclusive).
        width: image width (default 416).
        height: image height (default 416).

    Returns:
        idx: [K] flattened indices (within bounds).
        xy_valid: [K, 2] filtered coordinates.
        mask: boolean mask over input xy marking valid entries.
    """
    xy = xy.round().long()
    mask = (
        (xy[..., 0] >= x_min)
        & (xy[..., 0] <= x_max)
        & (xy[..., 1] >= y_min)
        & (xy[..., 1] <= y_max)
        & (xy[..., 0] >= 0)
        & (xy[..., 0] < width)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] < height)
    )

    xy_valid = xy[mask]
    idx = xy_valid[..., 1] * width + xy_valid[..., 0]
    return idx, xy_valid, mask


def _check_idx(name: str, idx: torch.Tensor, size: int) -> None:
    """Sanity-check index tensor (dtype & range)."""
    assert idx.dtype == torch.long, f"{name}: dtype must be long"
    if idx.numel():
        mn = int(idx.min().item())
        mx = int(idx.max().item())
        assert 0 <= mn <= mx < size, f"{name}: range [{mn}, {mx}] out of [0, {size - 1}]"

def _search_nearest_pt_torch(
    prev_points: torch.Tensor,   # [N, 2]
    now_points: torch.Tensor,    # [M, 2]
    now_confs: torch.Tensor,     # [M]
    max_dist: float = 5 * 2**0.5,
    lambda_dist: float = 1.0,
):
    """
    ...
    Returns:
        now_id_list: indices into now_points for valid matches
        valid_index: indices into prev_points that obtained a match
        matched_confs: confidence values of matched now_points [len(now_id_list)]
    """

    if prev_points.numel() == 0 or now_points.numel() == 0:
        empty_idx = torch.empty(0, dtype=torch.long, device=prev_points.device)
        empty_conf = now_confs.new_empty(0)
        return empty_idx, empty_idx, empty_conf, now_confs.new_tensor(0.0)

    dists = torch.cdist(prev_points, now_points)
    min_dists = dists.amin(dim=1)

    dists.mul_(-lambda_dist).add_(now_confs.unsqueeze(0))
    min_allowed = now_confs.unsqueeze(0) - float(lambda_dist * max_dist)
    dists.masked_fill_(dists < min_allowed, -INF)

    best_score, best_idx = dists.max(dim=1)
    valid_mask = best_score > -INF * 0.5

    now_id_list = best_idx[valid_mask]
    valid_index = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    matched_confs = now_confs[now_id_list]

    count = valid_mask.sum().clamp_min(1)
    valid_min_dists_mean = min_dists.masked_fill(~valid_mask, 0.0).sum() / count

    return now_id_list, valid_index, matched_confs, valid_min_dists_mean


class DetRefine(nn.Module):
    def __init__(self, search_radius, W):
        super().__init__()
        self.search_radius = search_radius
        self.W = W

    def forward(
        self,
        # previous frame info
        prev_kpts1,           # [N,2]
        prev_subref1,         # [N,2]
        prev_f1_2_center,     # [N,2]
        prev_m_bids,          # [N]

        # current frame info (sorted by conf)
        mkpts0_f,             # [M,2]
        mkpts1_f,             # [M,2]
        mconf_f_sorted,       # [M]

        # 7x7 unfold patches (stride=1) from image0 & image1
        feat_f0_unfold_st1,   # [B, L0, 49, Cf]
        feat_f1_unfold_st1,   # [B, L1, W^2, Cf]

        # bounds for _flat_idx_from_xy (computed outside)
        x_min, y_min, x_max, y_max
    ):
        """
        This is a pure Tensor-level implementation of the refinement block.
        """

        # -----------------------------
        # 1) previous centers → index in 1/2-res grid
        # -----------------------------
        idx0, prev_c0, mask0 = _flat_idx_from_xy(
            prev_f1_2_center, x_min, y_min, x_max, y_max
        )
        prev_kpts1 = prev_kpts1[mask0]
        prev_subref1 = prev_subref1[mask0]
        prev_m_bids = prev_m_bids[mask0]

        # -----------------------------
        # 2) nearest-neighbor search prev → now
        # -----------------------------
        now_id_list, valid_index, matched_confs, valid_min_dists_mean = _search_nearest_pt_torch(
            prev_kpts1, mkpts0_f, mconf_f_sorted, max_dist=self.search_radius
        )
        mk0_sel = mkpts0_f[now_id_list]
        mk1_sel = mkpts1_f[now_id_list]

        diff = (mk1_sel - mk0_sel) * 0.5
        diff_int = torch.round(diff).to(torch.int)

        # -----------------------------
        # 3) refine previous coarse centers
        # -----------------------------
        prev_c0 = prev_c0[valid_index]
        prev_subref1 = prev_subref1[valid_index]
        prev_m_bids = prev_m_bids[valid_index]

        mk1_2_center = prev_c0 + diff_int  # new centers

        # -----------------------------
        # 4) compute valid bounds from mkpts1_f
        # -----------------------------

        xy = mkpts1_f // 2
        x2_min = xy[:,0].min()
        y2_min = xy[:,1].min()
        x2_max = xy[:,0].max()
        y2_max = xy[:,1].max()

        idx1, new_c1, mask1 = _flat_idx_from_xy(
            mk1_2_center,
            x2_min, y2_min,
            x2_max, y2_max,
        )

        # mask both sides

        
        idx0 = idx0[valid_index][mask1]
        prev_m_bids = prev_m_bids[mask1]
        matched_confs = matched_confs[mask1]
        diff = diff[mask1]
        mk0_sel = mk0_sel[mask1]
        mk1_sel = mk1_sel[mask1]
        prev_subref1_sel = prev_subref1[mask1]

        # -----------------------------
        # 5) gather new 7x7 patches
        # -----------------------------
        feat0 = feat_f0_unfold_st1[prev_m_bids, idx0.long()]  # [K,49,C]
        feat1 = feat_f1_unfold_st1[prev_m_bids, idx1.long()]  # [K,W^2,C]

        # -----------------------------
        # 6) re-center 7x7 → 5x5
        # -----------------------------
        feat0_5x5 = _project_7x7_to_5x5(feat0, prev_subref1_sel)

        return (
            diff * 2,               # original-resolution displacement
            mk0_sel, mk1_sel,       # refined matching points
            matched_confs,          # confidence of matched refined pairs
            prev_m_bids,            # batch ids
            idx0.long(), idx1.long(),  # i_ids / j_ids
            prev_subref1_sel,       # sub-pixel offsets
            feat0_5x5,
            feat1,  
            valid_min_dists_mean,   # average nearest neighbor distance for valid matches
        )



class JamMa(nn.Module):
    def __init__(self, config, profiler=None):
        super().__init__()
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.d_model_c = self.config["coarse"]["d_model"]
        self.d_model_f = self.config["fine"]["d_model"]

        # coarse-level
        self.kenc = KeypointEncoder_wo_score(
            self.d_model_c, [32, 64, 128, self.d_model_c]
        )
        self.joint_mamba = JointMamba(
            self.d_model_c,
            4,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            profiler=self.profiler,
        )
        self.coarse_matching = CoarseMatching(config["match_coarse"], self.profiler)

        # FPN-style feature fusion (1/8 -> 1/4 -> 1/2)
        self.act = nn.GELU()
        dim = [256, 128, 64]
        self.up2 = up_conv4(dim[0], dim[1], dim[1])  # 1/8 -> 1/4
        self.conv7a = nn.Conv2d(2 * dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.conv7b = nn.Conv2d(dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.up3 = up_conv4(dim[1], dim[2], dim[2])  # 1/4 -> 1/2
        self.conv8a = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)
        self.conv8b = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)

        # fine-level MLP-Mixer encoder (window-based)
        W = self.config["fine_window_size"]
        self.fine_enc = nn.ModuleList(
            [MLPMixerEncoderLayer(2 * W**2, 64) for _ in range(4)]
        )
        self.fine_matching = FineSubMatching(config, self.profiler)
        self.use_det = self.config["det"]["use_det"]
        self.search_radius = self.config["det"]["search_radius"]

        self.detref = DetRefine(
            search_radius=self.search_radius,
            W=W,
        )

    # ------------------------ Coarse Matching ------------------------ #
    @torch.compile
    def coarse_match(self, data: dict) -> None:
        """Perform coarse-level feature interaction and matching."""
        desc0 = data["feat_8_0"].flatten(2, 3)
        desc1 = data["feat_8_1"].flatten(2, 3)
        kpts0 = data["grid_8"]
        kpts1 = data["grid_8"]

        # keypoint normalization
        kpts0 = normalize_keypoints(kpts0, data["imagec_0"].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data["imagec_1"].shape[-2:])

        kpts0 = kpts0.transpose(1, 2)
        kpts1 = kpts1.transpose(1, 2)

        # add position encoding
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)

        data.update({"feat_8_0": desc0, "feat_8_1": desc1})

        # joint feature interaction
        with self.profiler.profile("coarse interaction"):
            self.joint_mamba(data)

        # coarse matching
        mask_c0 = mask_c1 = None
        if "mask0" in data:
            mask_c0 = data["mask0"].flatten(-2)
            mask_c1 = data["mask1"].flatten(-2)

        with self.profiler.profile("coarse matching"):
            self.coarse_matching(
                data["feat_8_0"].transpose(1, 2),
                data["feat_8_1"].transpose(1, 2),
                data,
                mask_c0=mask_c0,
                mask_c1=mask_c1,
            )

    # ------------------------ FPN Fusion ------------------------ #
    @torch.compile
    def inter_fpn(self, feat_8: torch.Tensor, feat_4: torch.Tensor) -> torch.Tensor:
        """Intermediate FPN: merge 1/8 and 1/4 features then upsample to 1/2.

        Args:
            feat_8: [2B, C8, H8, W8]
            feat_4: [2B, C4, H4, W4]

        Returns:
            feat_2: [2B, C2, H2, W2] (1/2 resolution)
        """
        # 1/8 -> 1/4 and fuse with 1/4 features
        d2 = self.up2(feat_8)
        d2 = self.act(self.conv7a(torch.cat([feat_4, d2], dim=1)))
        feat_4 = self.act(self.conv7b(d2))

        # 1/4 -> 1/2
        d1 = self.up3(feat_4)
        d1 = self.act(self.conv8a(d1))
        feat_2 = self.conv8b(d1)
        return feat_2

    # ------------------------ Fine Preprocess ------------------------ #

    @torch.compile
    def fine_preprocess(
        self, data: dict, profiler: PassThroughProfiler
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare fine-level features and windows for matching.

        Returns:
            feat_f0_unfold: [M, W^2, C_f]
            feat_f1_unfold: [M, W^2, C_f]
        """
        data["resolution1"] = 8
        stride = data["resolution1"] // self.config["resolution"][1]
        W = self.config["fine_window_size"]

        feat_8 = torch.cat(
            [data["feat_8_0"], data["feat_8_1"]], dim=0
        ).view(2 * data["bs"], data["c"], data["h_8"], -1)
        feat_4 = torch.cat([data["feat_4_0"], data["feat_4_1"]], dim=0)

        if data["b_ids"].shape[0] == 0:
            device = feat_4.device
            empty = torch.empty(0, W**2, self.d_model_f, device=device)
            return empty, empty

        # feature fusion (1/8 + 1/4 -> 1/2)
        feat_f = self.inter_fpn(feat_8, feat_4)
        feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        data.update({"hw0_f": feat_f0.shape[2:], "hw1_f": feat_f1.shape[2:]})

        # 1. unfold all local windows
        pad = 0 if W % 2 == 0 else W // 2

        # image 0
        feat_f0_unfold = F.unfold(
            feat_f0, kernel_size=(W, W), stride=stride, padding=pad
        )
        feat_f0_unfold = rearrange(
            feat_f0_unfold, "n (c ww) l -> n l ww c", ww=W**2
        )

        # image 1
        feat_f1_unfold = F.unfold(
            feat_f1, kernel_size=(W, W), stride=stride, padding=pad
        )
        feat_f1_unfold = rearrange(
            feat_f1_unfold, "n (c ww) l -> n l ww c", ww=W**2
        )  # [B, H_f/stride * W_f/stride, W^2, C]

        # 2. select only the locations corresponding to coarse matches
        feat_f0_unfold = feat_f0_unfold[data["b_ids"], data["i_ids"]]  # [M, W^2, C_f]
        feat_f1_unfold = feat_f1_unfold[data["b_ids"], data["j_ids"]]  # [M, W^2, C_f]

        # additional unfold for detection mode (stride=1)
        feat_f0_unfold_st1 = F.unfold(
            feat_f0, kernel_size=(7, 7), stride=1, padding=3
        )
        feat_f0_unfold_st1 = rearrange(
            feat_f0_unfold_st1, "n (c ww) l -> n l ww c", ww=7**2
        )
        feat_f1_unfold_st1 = F.unfold(
            feat_f1, kernel_size=(W, W), stride=1, padding=pad
        )
        feat_f1_unfold_st1 = rearrange(
            feat_f1_unfold_st1, "n (c ww) l -> n l ww c", ww=W**2
        )
        data.update(
            {
                "feat_f0_unfold_st1": feat_f0_unfold_st1,
                "feat_f1_unfold_st1": feat_f1_unfold_st1,
            }
        )

        # concat features from two views and apply MLP-Mixer
        feat_f_cat = torch.cat([feat_f0_unfold, feat_f1_unfold], dim=1).transpose(
            1, 2
        )  # [M, C_f, 2W^2]

        for layer in self.fine_enc:
            feat_f_cat = layer(feat_f_cat)

        feat_f0_unfold, feat_f1_unfold = (
            feat_f_cat[:, :, : W**2],
            feat_f_cat[:, :, W**2 :],
        )
        return feat_f0_unfold, feat_f1_unfold

    # ------------------------ Forward ------------------------ #
    def forward(self, data: dict, mode: str = "test") -> None:
        self.mode = mode

        data.update(
            {
                "hw0_i": data["imagec_0"].shape[2:],
                "hw1_i": data["imagec_1"].shape[2:],
                "hw0_c": [data["h_8"], data["w_8"]],
                "hw1_c": [data["h_8"], data["w_8"]],
            }
        )

        # 1. coarse-level matching
        self.coarse_match(data)

        # 2. fine-level matching and refinement
        with self.profiler.profile("fine matching"):
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
                data, self.profiler
            )

            # 2.1 base fine-level matching
            self.fine_matching(
                feat_f0_unfold.transpose(1, 2),
                feat_f1_unfold.transpose(1, 2),
                data,
                phase=0,
            )

            # 2.2 detection-based refinement using previous frame
            if self.use_det and "prev_data" in data:
                prev_data = data["prev_data"]

                # only if previous image_1 equals current image_0
                if prev_data["image_idB"] == data["image_idA"]:
                    # sort previous matches by confidence
                    prev_sort_idx = torch.topk(
                        prev_data["mconf_f"],
                        len(prev_data["mconf_f"]),
                        dim=0,
                    ).indices
                    prev_kpts1 = prev_data["mkpts1_f"][prev_sort_idx]
                    prev_subref1 = prev_data["mkpts1_subref"][prev_sort_idx]
                    prev_f1_2_center = (
                        prev_data["mkpts1_f1_fine"][prev_sort_idx]
                        + prev_data["mkpts1_f1_window"][prev_sort_idx]
                        - 2
                    )


                    # sort current matches by confidence
                    sort_idx = torch.topk(
                        data["mconf_f"], len(data["mconf_f"]), dim=0
                    ).indices
                    now_mkpts0_f = data["mkpts0_f"][sort_idx]
                    now_mkpts1_f = data["mkpts1_f"][sort_idx]

                    now_confs_sorted = data["mconf_f"][sort_idx]

                    (
                        diff,
                        mk0_sel,
                        mk1_sel,
                        conf_sel,
                        new_b_ids,
                        new_i_ids,
                        new_j_ids,
                        prev_subref1_sel,
                        feat_f0_unfold_flex,
                        feat_f1_unfold_flex,
                        valid_min_dists_mean,
                    ) = self.detref(
                        prev_kpts1,
                        prev_subref1,
                        prev_f1_2_center,
                        prev_data["m_bids"],
                        now_mkpts0_f,
                        now_mkpts1_f,
                        now_confs_sorted,
                        data["feat_f0_unfold_st1"],
                        data["feat_f1_unfold_st1"],
                        3,3,413,413,  # bounds for previous coarse centers
                    )


                    feat_f_flex = torch.cat(
                        [feat_f0_unfold_flex, feat_f1_unfold_flex], dim=1
                    ).transpose(1, 2)
                    for layer in self.fine_enc:
                        feat_f_flex = layer(feat_f_flex)

                    W = self.config["fine_window_size"]
                    feat_f0_unfold_flex, feat_f1_unfold_flex = (
                        feat_f_flex[:, :, : W**2],
                        feat_f_flex[:, :, W**2 :],
                    )

                    data.update({"diff": diff})
                    data.update(
                        {
                            "diff_points": {
                                "0": mk0_sel,
                                "1": mk1_sel,
                            },
                            "diff_points_conf": conf_sel,
                            "diff_points_valid_mean_dist": valid_min_dists_mean,
                        }
                    )
                    data.update(
                        {
                            "new_b_ids": new_b_ids,
                            "new_i_ids": new_i_ids,
                            "new_j_ids": new_j_ids,
                            "prev_subref1": prev_subref1_sel,
                        }
                    )
                    # detection-phase fine matching
                    self.fine_matching(
                        feat_f0_unfold_flex.transpose(1, 2),
                        feat_f1_unfold_flex.transpose(1, 2),
                        data,
                        phase=1,
                    )
