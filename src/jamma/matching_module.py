import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from loguru import logger

from src.jamma.compile_utils import maybe_compile

INF = 1e9


# -------------------- Utility functions -------------------- #

def mask_border(m: torch.Tensor, b: int, v) -> None:
    """Mask borders of a 5D tensor with value `v`.

    Args:
        m: tensor of shape [N, H0, W0, H1, W1]
        b: border width
        v: value to set
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(
    m: torch.Tensor,
    bd: int,
    v,
    p_m0: torch.Tensor,
    p_m1: torch.Tensor,
) -> None:
    """Mask borders while respecting padded valid regions.

    Args:
        m: tensor of shape [N, H0, W0, H1, W1]
        bd: border width
        v: value to set
        p_m0, p_m1: padding masks for image0 and image1 [N, H, W]
    """
    if bd <= 0:
        return

    # top/left borders (shared for all examples)
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    # compute valid heights/widths per batch for bottom/right borders
    h0s = p_m0.sum(1).max(-1)[0].int()
    w0s = p_m0.sum(-1).max(-1)[0].int()
    h1s = p_m1.sum(1).max(-1)[0].int()
    w1s = p_m1.sum(-1).max(-1)[0].int()

    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0: torch.Tensor, p_m1: torch.Tensor) -> torch.Tensor:
    """Compute the maximum possible number of candidate matches in a batch.

    Args:
        p_m0, p_m1: padded masks [N, H, W]

    Returns:
        Scalar tensor: sum over batch of min(valid_area_img0, valid_area_img1).
    """
    h0s = p_m0.sum(1).max(-1)[0]
    w0s = p_m0.sum(-1).max(-1)[0]
    h1s = p_m1.sum(1).max(-1)[0]
    w1s = p_m1.sum(-1).max(-1)[0]

    area0 = h0s * w0s
    area1 = h1s * w1s
    max_cand = torch.min(torch.stack([area0, area1], dim=-1), dim=-1)[0].sum()
    return max_cand


def generate_random_mask(n: int, num_true: int, device=None) -> torch.Tensor:
    """Generate a boolean mask with exactly `num_true` True values.

    Args:
        n: total length
        num_true: number of True entries
        device: tensor device

    Returns:
        mask: [n] bool
    """
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    indices = torch.randperm(n, device=device)[:num_true]
    mask[indices] = True
    return mask


def _idx_to_xy(idx: torch.Tensor, width: int) -> torch.Tensor:
    """Convert flattened indices to (x, y) coordinates.

    Args:
        idx: tensor of indices (...,)
        width: image width

    Returns:
        coords: (..., 2) where coords[..., 0] = x, coords[..., 1] = y
    """
    x = idx % width
    y = torch.div(idx, width, rounding_mode="trunc")
    return torch.stack([x, y], dim=-1)


# -------------------- Coarse Matching -------------------- #

class CoarseMatching(nn.Module):
    def __init__(self, config, profiler):
        super().__init__()
        self.config = config

        d_model = 256
        self.thr = config["thr"]
        self.use_sm = config["use_sm"]
        self.inference = config["inference"]
        self.border_rm = config["border_rm"]

        self.final_proj = nn.Linear(d_model, d_model, bias=True)
        self.temperature = config["dsmax_temperature"]
        self.inv_sqrt_d = d_model ** -0.5  # precompute for normalization
        self.profiler = profiler

    def forward(
        self,
        feat_c0: torch.Tensor,
        feat_c1: torch.Tensor,
        data: dict,
        mask_c0: torch.Tensor = None,
        mask_c1: torch.Tensor = None,
    ) -> None:
        # final linear projection
        feat_c0 = self.final_proj(feat_c0)
        feat_c1 = self.final_proj(feat_c1)

        # normalize (scaled L2-like normalization)
        feat_c0 = feat_c0 * self.inv_sqrt_d
        feat_c1 = feat_c1 * self.inv_sqrt_d

        # similarity matrix [B, L0, L1]
        #sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        sim_matrix = torch.bmm(feat_c0, feat_c1.transpose(1, 2)) / self.temperature

        # optional masking (e.g., valid regions only)
        if mask_c0 is not None and mask_c1 is not None:
            #valid = (mask_c0[..., None] * mask_c1[:, None]).bool()
            #sim_matrix.masked_fill_(~valid, -INF)

            valid = mask_c0[..., None] & mask_c1[:, None, :]
            sim_matrix.masked_fill_(~valid, -INF)


        if self.inference:
            data.update(**self.get_coarse_match_inference(sim_matrix, data))
        else:
            conf_matrix_0_to_1 = F.softmax(sim_matrix, dim=2)
            conf_matrix_1_to_0 = F.softmax(sim_matrix, dim=1)
            data.update(
                {
                    "conf_matrix_0_to_1": conf_matrix_0_to_1,
                    "conf_matrix_1_to_0": conf_matrix_1_to_0,
                }
            )
            data.update(
                **self.get_coarse_match_training(
                    conf_matrix_0_to_1, conf_matrix_1_to_0, data
                )
            )

    @torch.no_grad()
    def get_coarse_match_training(
        self,
        conf_matrix_0_to_1: torch.Tensor,
        conf_matrix_1_to_0: torch.Tensor,
        data: dict,
    ) -> dict:
        """Coarse matching logic for training.

        Produces a set of candidate matches and (optionally) augments with
        ground-truth matches for training stability.
        """
        h0c, w0c = data["hw0_c"]
        h1c, w1c = data["hw1_c"]
        axes_lengths = {"h0c": h0c, "w0c": w0c, "h1c": h1c, "w1c": w1c}
        device = conf_matrix_0_to_1.device

        # confidence thresholding:
        # {(nearest neighbor for 0->1) U (nearest neighbor for 1->0)}
        mask = torch.logical_or(
            (conf_matrix_0_to_1 > self.thr)
            & (conf_matrix_0_to_1 == conf_matrix_0_to_1.max(dim=2, keepdim=True)[0]),
            (conf_matrix_1_to_0 > self.thr)
            & (conf_matrix_1_to_0 == conf_matrix_1_to_0.max(dim=1, keepdim=True)[0]),
        )

        mask = rearrange(
            mask,
            "b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c",
            **axes_lengths,
        )
        if "mask0" not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(
                mask, self.border_rm, False, data["mask0"], data["mask1"]
            )
        mask = rearrange(
            mask,
            "b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)",
            **axes_lengths,
        )

        # find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)

        mconf = torch.maximum(
            conf_matrix_0_to_1[b_ids, i_ids, j_ids],
            conf_matrix_1_to_0[b_ids, i_ids, j_ids],
        )

        # random sampling of training samples for fine-level module
        if self.training:
            if "mask0" not in data:
                num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data["mask0"], data["mask1"]
                )

            num_matches_train = int(
                num_candidates_max * self.config["train_coarse_percent"]
            )
            num_matches_pred = len(b_ids)
            train_pad_num_gt_min = self.config["train_pad_num_gt_min"]
            assert (
                train_pad_num_gt_min < num_matches_train
            ), "min-num-gt-pad should be less than num-train-matches"

            if num_matches_pred <= num_matches_train - train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - train_pad_num_gt_min,),
                    device=device,
                )

            gt_pad_len = max(
                num_matches_train - num_matches_pred,
                train_pad_num_gt_min,
            )
            gt_pad_indices = torch.randint(
                len(data["spv_b_ids"]),
                (gt_pad_len,),
                device=device,
            )
            mconf_gt = torch.zeros(len(data["spv_b_ids"]), device=device)

            b_ids = torch.cat([b_ids[pred_indices], data["spv_b_ids"][gt_pad_indices]])
            i_ids = torch.cat([i_ids[pred_indices], data["spv_i_ids"][gt_pad_indices]])
            j_ids = torch.cat([j_ids[pred_indices], data["spv_j_ids"][gt_pad_indices]])
            mconf = torch.cat([mconf[pred_indices], mconf_gt[gt_pad_indices]])

        coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        scale = data["hw0_i"][0] / data["hw0_c"][0]
        if "scale0" in data:
            scale0 = scale * data["scale0"][b_ids]
            scale1 = scale * data["scale1"][b_ids]
        else:
            scale0 = scale
            scale1 = scale

        mkpts0_c = _idx_to_xy(i_ids, data["hw0_c"][1]) * scale0
        mkpts1_c = _idx_to_xy(j_ids, data["hw1_c"][1]) * scale1

        valid_mask = mconf != 0

        coarse_matches.update(
            {
                "gt_mask": mconf == 0,
                "m_bids": b_ids[valid_mask],
                "mkpts0_c": mkpts0_c[valid_mask],
                "mkpts1_c": mkpts1_c[valid_mask],
                "mkpts0_c_train": mkpts0_c,
                "mkpts1_c_train": mkpts1_c,
                "mconf": mconf[valid_mask],
            }
        )

        return coarse_matches

    @torch.no_grad()
    def get_coarse_match_inference(
        self,
        sim_matrix: torch.Tensor,
        data: dict,
    ) -> dict:
        """Coarse matching logic for inference."""
        h0c, w0c = data["hw0_c"]
        h1c, w1c = data["hw1_c"]
        axes_lengths = {"h0c": h0c, "w0c": w0c, "h1c": h1c, "w1c": w1c}

        conf_matrix = F.softmax(sim_matrix, dim=2) if self.use_sm else sim_matrix
        mask = (conf_matrix > self.thr) & (
            conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]
        )

        conf_matrix = F.softmax(sim_matrix, dim=1) if self.use_sm else sim_matrix
        mask = torch.logical_or(
            mask,
            (conf_matrix > self.thr)
            & (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0]),
        )

        mask = rearrange(
            mask,
            "b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c",
            **axes_lengths,
        )
        if "mask0" not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(
                mask, self.border_rm, False, data["mask0"], data["mask1"]
            )
        mask = rearrange(
            mask,
            "b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)",
            **axes_lengths,
        )

        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)
        mconf = sim_matrix[b_ids, i_ids, j_ids]

        coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        scale = data["hw0_i"][0] / data["hw0_c"][0]
        if "scale0" in data:
            scale0 = scale * data["scale0"][b_ids]
            scale1 = scale * data["scale1"][b_ids]
        else:
            scale0 = scale
            scale1 = scale

        mkpts0_c = _idx_to_xy(i_ids, data["hw0_c"][1]) * scale
        mkpts1_c = _idx_to_xy(j_ids, data["hw1_c"][1]) * scale
        mkpts0_c_origin = _idx_to_xy(i_ids, data["hw0_c"][1]) * scale0
        mkpts1_c_origin = _idx_to_xy(j_ids, data["hw1_c"][1]) * scale1

        coarse_matches.update(
            {
                "mconf": mconf,
                "m_bids": b_ids,
                "mkpts0_c": mkpts0_c,
                "mkpts1_c": mkpts1_c,
                "mkpts0_c_origin": mkpts0_c_origin,
                "mkpts1_c_origin": mkpts1_c_origin,
            }
        )

        return coarse_matches


# -------------------- Fine & Sub-pixel Matching -------------------- #

class FineSubMatching(nn.Module):
    """Fine-level and sub-pixel matching."""

    def __init__(self, config, profiler):
        super().__init__()
        self.temperature = config["fine"]["dsmax_temperature"]
        self.W_f = config["fine_window_size"]
        self.inference = config["fine"]["inference"]

        dim_f = 64
        self.fine_thr = config["fine"]["thr"]
        self.fine_proj = nn.Linear(dim_f, dim_f, bias=False)
        self.subpixel_mlp = nn.Sequential(
            nn.Linear(2 * dim_f, 2 * dim_f, bias=False),
            nn.ReLU(),
            nn.Linear(2 * dim_f, 4, bias=False),
        )
        self.fine_spv_max = 500
        self.profiler = profiler
        self.use_det = config["det"]["use_det"]
        self.fine_det_thr = config["det"]["fine_thr"]
        self.inv_sqrt_df = dim_f ** -0.5  # precompute for normalization

    def forward(
        self,
        feat_f0_unfold: torch.Tensor,
        feat_f1_unfold: torch.Tensor,
        data: dict,
        phase: int = 0,
    ) -> None:
        """Forward pass for fine-level matching.

        Args:
            feat_f0_unfold: [M, W^2, C]
            feat_f1_unfold: [M, W^2, C]
            data: data dict with coarse matches and meta info
            phase: 0 = normal fine matching, 1 = detection mode refinement
        """
        M, WW, C = feat_f0_unfold.shape
        W_f = self.W_f

        if M == 0:
            assert not self.training, "M is always > 0 during training (see coarse matching)."
            logger.warning("No matches found in coarse-level.")

            device = feat_f0_unfold.device
            if self.inference:
                data.update(
                    {
                        "mkpts0_f": data["mkpts0_c"],
                        "mkpts1_f": data["mkpts1_c"],
                        "mconf_f": torch.zeros(0, device=device),
                    }
                )
            else:
                data.update(
                    {
                        "mkpts0_f": data["mkpts0_c"],
                        "mkpts1_f": data["mkpts1_c"],
                        "mconf_f": torch.zeros(0, device=device),
                        "mkpts0_f_train": data["mkpts0_c_train"],
                        "mkpts1_f_train": data["mkpts1_c_train"],
                        "conf_matrix_fine": torch.zeros(
                            1, W_f * W_f, W_f * W_f, device=device
                        ),
                        "b_ids_fine": torch.zeros(0, device=device),
                        "i_ids_fine": torch.zeros(0, device=device),
                        "j_ids_fine": torch.zeros(0, device=device),
                    }
                )
            return

        feat_f0 = self.fine_proj(feat_f0_unfold) * self.inv_sqrt_df
        feat_f1 = self.fine_proj(feat_f1_unfold) * self.inv_sqrt_df

        #sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0, feat_f1) / self.temperature
        sim_matrix = torch.bmm(feat_f0, feat_f1.transpose(1, 2)) / self.temperature

        conf_matrix_fine = F.softmax(sim_matrix, dim=1) * F.softmax(sim_matrix, dim=2)
        sim_softmax_0_to_1 = F.softmax(sim_matrix, dim=2)

        if phase == 0:
            data.update(
                **self.get_fine_sub_match(
                    conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data
                )
            )
        else:
            if self.use_det:
                data.update(
                    **self.get_fine_sub_match_det(
                        conf_matrix_fine,
                        feat_f0_unfold,
                        feat_f1_unfold,
                        data,
                        sim_softmax_0_to_1,
                    )
                )

    @maybe_compile
    def get_fine_sub_match(
        self,
        conf_matrix_fine: torch.Tensor,
        feat_f0_unfold: torch.Tensor,
        feat_f1_unfold: torch.Tensor,
        data: dict,
    ) -> dict:
        """Standard fine-level + sub-pixel matching."""
        with torch.no_grad():
            W_f = self.W_f
            b_ids_c, i_ids_c, j_ids_c = data["b_ids"], data["i_ids"], data["j_ids"]

            mask_conf = conf_matrix_fine > self.fine_thr
            if mask_conf.sum() == 0:
                mask_conf[0, 0, 0] = 1
                conf_matrix_fine[0, 0, 0] = 1

            mask = mask_conf & (
                conf_matrix_fine
                == conf_matrix_fine.amax(dim=[1, 2], keepdim=True)
            )

            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix_fine[b_ids, i_ids, j_ids]

            scale_f_c = data["hw0_f"][0] // data["hw0_c"][0]
            mkpts0_c_scaled_to_f = _idx_to_xy(
                i_ids_c, data["hw0_c"][1]
            ) * scale_f_c
            mkpts1_c_scaled_to_f = _idx_to_xy(
                j_ids_c, data["hw1_c"][1]
            ) * scale_f_c

            updated_b_ids = b_ids_c[b_ids]

            scale = data["hw0_i"][0] / data["hw0_f"][0]
            if "scale0" in data:
                scale0 = scale * data["scale0"][updated_b_ids]
                scale1 = scale * data["scale1"][updated_b_ids]
            else:
                scale0 = scale
                scale1 = scale

            mkpts0_f_window = _idx_to_xy(i_ids, W_f)
            mkpts1_f_window = _idx_to_xy(j_ids, W_f)

        sub_ref = self.subpixel_mlp(
            torch.cat(
                [feat_f0_unfold[b_ids, i_ids], feat_f1_unfold[b_ids, j_ids]],
                dim=-1,
            )
        )
        sub_ref0, sub_ref1 = torch.chunk(sub_ref, 2, dim=1)
        sub_ref0 = torch.tanh(sub_ref0.squeeze(1)) * 0.5
        sub_ref1 = torch.tanh(sub_ref1.squeeze(1)) * 0.5

        pad = 0 if W_f % 2 == 0 else W_f // 2

        mkpts0_f1 = (
            mkpts0_f_window + mkpts0_c_scaled_to_f[b_ids] - pad
        ) * scale
        mkpts1_f1 = (
            mkpts1_f_window + mkpts1_c_scaled_to_f[b_ids] - pad
        ) * scale

        mkpts0_f1_origin = (
            mkpts0_f_window + mkpts0_c_scaled_to_f[b_ids] - pad
        ) * scale0
        mkpts1_f1_origin = (
            mkpts1_f_window + mkpts1_c_scaled_to_f[b_ids] - pad
        ) * scale1

        mkpts0_f_train = mkpts0_f1 + sub_ref0 * scale
        mkpts1_f_train = mkpts1_f1 + sub_ref1 * scale
        mkpts0_f_train_origin = mkpts0_f1_origin + sub_ref0 * scale0
        mkpts1_f_train_origin = mkpts1_f1_origin + sub_ref1 * scale1

        mkpts0_f = mkpts0_f_train.detach()
        mkpts1_f = mkpts1_f_train.detach()
        mkpts0_f_origin = mkpts0_f_train_origin.detach()
        mkpts1_f_origin = mkpts1_f_train_origin.detach()

        valid_mask = mconf != 0

        sub_pixel_matches = {
            "m_bids": b_ids_c[b_ids[valid_mask]],
            "mkpts0_f1": mkpts0_f1[valid_mask],
            "mkpts1_f1": mkpts1_f1[valid_mask],
            "mkpts0_f": mkpts0_f[valid_mask],
            "mkpts1_f": mkpts1_f[valid_mask],
            "mconf_f": mconf[valid_mask],
            "mkpts0_f1_fine": mkpts0_c_scaled_to_f[b_ids][valid_mask],
            "mkpts1_f1_fine": mkpts1_c_scaled_to_f[b_ids][valid_mask],
            "mkpts0_f1_window": mkpts0_f_window[valid_mask],
            "mkpts1_f1_window": mkpts1_f_window[valid_mask],
            "mkpts0_subref": sub_ref0[valid_mask],
            "mkpts1_subref": sub_ref1[valid_mask],
            "mkpts0_f1_origin": mkpts0_f1_origin[valid_mask],
            "mkpts1_f1_origin": mkpts1_f1_origin[valid_mask],
            "mkpts0_f_origin": mkpts0_f_origin[valid_mask],
            "mkpts1_f_origin": mkpts1_f_origin[valid_mask],
        }

        if not self.inference:
            b_ids_all = data["b_ids"]
            device = b_ids_all.device
            if self.fine_spv_max is None or self.fine_spv_max > len(b_ids_all):
                sub_pixel_matches.update(
                    {
                        "mkpts0_f_train": mkpts0_f_train,
                        "mkpts1_f_train": mkpts1_f_train,
                        "b_ids_fine": b_ids_all,
                        "i_ids_fine": data["i_ids"],
                        "j_ids_fine": data["j_ids"],
                        "conf_matrix_fine": conf_matrix_fine,
                    }
                )
            else:
                train_mask = generate_random_mask(
                    len(b_ids_all), self.fine_spv_max, device=device
                )
                sub_pixel_matches.update(
                    {
                        "mkpts0_f_train": mkpts0_f_train,
                        "mkpts1_f_train": mkpts1_f_train,
                        "b_ids_fine": b_ids_all[train_mask],
                        "i_ids_fine": data["i_ids"][train_mask],
                        "j_ids_fine": data["j_ids"][train_mask],
                        "conf_matrix_fine": conf_matrix_fine[train_mask],
                    }
                )

        return sub_pixel_matches

    @maybe_compile
    def get_fine_sub_match_det(
        self,
        conf_matrix_fine: torch.Tensor,
        feat_f0_unfold: torch.Tensor,
        feat_f1_unfold: torch.Tensor,
        data: dict,
        sim_matrix: torch.Tensor,
    ) -> dict:
        """Fine-level + sub-pixel matching in detection mode."""
        with torch.no_grad():
            W_f = self.W_f
            b_ids_c = data["new_b_ids"]
            diff = data["diff"]
            diff_points = data["diff_points"]
            diff_points_conf = data['diff_points_conf']

            N, I, J = conf_matrix_fine.shape
            assert I >= 13, f"fine window dim I={I} must be > 12"

            conf_row = conf_matrix_fine[:, 12, :]
            sim_row = sim_matrix[:, 12, :]

            _, j_top = sim_row.max(dim=1)
            conf_at_j = sim_row[torch.arange(N, device=sim_row.device), j_top]
            mask_valid = conf_at_j > self.fine_det_thr

            mask = torch.zeros_like(conf_matrix_fine, dtype=torch.bool)
            idx_valid = torch.arange(N, device=mask.device)[mask_valid]
            mask[idx_valid, 12, j_top[mask_valid]] = True

            mask_v_det, all_j_ids_det = mask.max(dim=2)
            b_ids_det, i_ids_det = torch.where(mask_v_det)
            j_ids_det = all_j_ids_det[b_ids_det, i_ids_det]

            B, I2, J2 = sim_matrix.shape
            assert (b_ids_det >= 0).all() and (b_ids_det < B).all()
            assert (i_ids_det == 12).all()
            assert (j_ids_det >= 0).all() and (j_ids_det < J2).all()

            mconf_det = sim_matrix[b_ids_det, i_ids_det, j_ids_det]
            assert torch.isfinite(mconf_det).all(), "mconf_det contains NaN/Inf"

            updated_b_ids = b_ids_c[b_ids_det]

            scale = data["hw0_i"][0] / data["hw0_f"][0]
            if "scale0" in data:
                scale0 = scale * data["scale0"][updated_b_ids]
                scale1 = scale * data["scale1"][updated_b_ids]
            else:
                scale0 = scale
                scale1 = scale

            mkpts0_c_scaled_to_f_det = _idx_to_xy(data["new_i_ids"], 416)
            mkpts1_c_scaled_to_f_det = _idx_to_xy(data["new_j_ids"], 416)

            mkpts0_f_window_det = _idx_to_xy(i_ids_det, W_f)
            mkpts1_f_window_det = _idx_to_xy(j_ids_det, W_f)

        pad = 0 if W_f % 2 == 0 else W_f // 2

        sub_ref_det = self.subpixel_mlp(
            torch.cat(
                [
                    feat_f0_unfold[b_ids_det, i_ids_det],
                    feat_f1_unfold[b_ids_det, j_ids_det],
                ],
                dim=-1,
            )
        )
        sub_ref0_det, sub_ref1_det = torch.chunk(sub_ref_det, 2, dim=1)
        sub_ref0_det = torch.tanh(sub_ref0_det.squeeze(1)) * 0.5
        sub_ref1_det = torch.tanh(sub_ref1_det.squeeze(1)) * 0.5

        sub_ref0_det = data["prev_subref1"][b_ids_det]

        mkpts0_f1_det = (
            mkpts0_f_window_det + mkpts0_c_scaled_to_f_det[b_ids_det] - pad
        ) * scale
        mkpts1_f1_det = (
            mkpts1_f_window_det + mkpts1_c_scaled_to_f_det[b_ids_det] - pad
        ) * scale

        mkpts0_f1_det_origin = (
            mkpts0_f_window_det + mkpts0_c_scaled_to_f_det[b_ids_det] - pad
        ) * scale0
        mkpts1_f1_det_origin = (
            mkpts1_f_window_det + mkpts1_c_scaled_to_f_det[b_ids_det] - pad
        ) * scale1

        mkpts0_f_train_det = mkpts0_f1_det + sub_ref0_det * scale
        mkpts1_f_train_det = mkpts1_f1_det + sub_ref1_det * scale
        mkpts0_f_train_det_origin = mkpts0_f1_det_origin + sub_ref0_det * scale0
        mkpts1_f_train_det_origin = mkpts1_f1_det_origin + sub_ref1_det * scale1

        mkpts0_f_det = mkpts0_f_train_det.detach()
        mkpts1_f_det = mkpts1_f_train_det.detach()
        mkpts0_f_det_origin = mkpts0_f_train_det_origin.detach()
        mkpts1_f_det_origin = mkpts1_f_train_det_origin.detach()

        valid_mask = mconf_det != 0

        diff_points["0"] = diff_points["0"][b_ids_det[valid_mask]]
        diff_points["1"] = diff_points["1"][b_ids_det[valid_mask]]
        diff_points_conf = diff_points_conf[b_ids_det[valid_mask]]

        num_valid = int(valid_mask.sum().item())
        num_target = len(data["mconf_f"])
        num_missing = num_target - num_valid

        mconf_det = mconf_det[valid_mask]
        mconf_det = mconf_det + diff_points_conf

        #mconf_det = mconf_det * diff_points_conf

        force = False
        if num_missing and force:
            conf_sorted_idx = torch.argsort(data["mconf_f"], descending=True)
            add_idx = conf_sorted_idx[:num_missing]

            add_m_bids = data["m_bids"][add_idx]
            add_mkpts0_f1 = data["mkpts0_f1"][add_idx]
            add_mkpts1_f1 = data["mkpts1_f1"][add_idx]
            add_mkpts0_f = data["mkpts0_f"][add_idx]
            add_mkpts1_f = data["mkpts1_f"][add_idx]
            add_mconf_f = data["mconf_f"][add_idx]
            add_mkpts0_f1_fine = data["mkpts0_f1_fine"][add_idx]
            add_mkpts1_f1_fine = data["mkpts1_f1_fine"][add_idx]
            add_mkpts0_f1_window = data["mkpts0_f1_window"][add_idx]
            add_mkpts1_f1_window = data["mkpts1_f1_window"][add_idx]
            add_mkpts0_subref = data["mkpts0_subref"][add_idx]
            add_mkpts1_subref = data["mkpts1_subref"][add_idx]
            add_mkpts0_f1_origin = data["mkpts0_f1_origin"][add_idx]
            add_mkpts1_f1_origin = data["mkpts1_f1_origin"][add_idx]
            add_mkpts0_f_origin = data["mkpts0_f_origin"][add_idx]
            add_mkpts1_f_origin = data["mkpts1_f_origin"][add_idx]

            sub_pixel_matches = {
                "m_bids": torch.cat([b_ids_c[b_ids_det[valid_mask]], add_m_bids]),
                "mkpts0_f1": torch.cat(
                    [mkpts0_f1_det[valid_mask], add_mkpts0_f1]
                ),
                "mkpts1_f1": torch.cat(
                    [mkpts1_f1_det[valid_mask], add_mkpts1_f1]
                ),
                "mkpts0_f": torch.cat(
                    [mkpts0_f_det[valid_mask], add_mkpts0_f]
                ),
                "mkpts1_f": torch.cat(
                    [mkpts1_f_det[valid_mask], add_mkpts1_f]
                ),
                "mconf_f": torch.cat(
                    [mconf_det, add_mconf_f]
                ),
                "diff": diff[b_ids_det[valid_mask]],
                "diff_points": diff_points,
                "diff_points_conf": diff_points_conf,
                "mkpts0_f1_fine": torch.cat(
                    [
                        mkpts0_c_scaled_to_f_det[b_ids_det][valid_mask],
                        add_mkpts0_f1_fine,
                    ]
                ),
                "mkpts1_f1_fine": torch.cat(
                    [
                        mkpts1_c_scaled_to_f_det[b_ids_det][valid_mask],
                        add_mkpts1_f1_fine,
                    ]
                ),
                "mkpts0_f1_window": torch.cat(
                    [mkpts0_f_window_det[valid_mask], add_mkpts0_f1_window]
                ),
                "mkpts1_f1_window": torch.cat(
                    [mkpts1_f_window_det[valid_mask], add_mkpts1_f1_window]
                ),
                "mkpts0_subref": torch.cat(
                    [sub_ref0_det[valid_mask], add_mkpts0_subref]
                ),
                "mkpts1_subref": torch.cat(
                    [sub_ref1_det[valid_mask], add_mkpts1_subref]
                ),
                "mkpts0_f1_origin": torch.cat(
                    [mkpts0_f1_det_origin[valid_mask], add_mkpts0_f1_origin]
                ),
                "mkpts1_f1_origin": torch.cat(
                    [mkpts1_f1_det_origin[valid_mask], add_mkpts1_f1_origin]
                ),
                "mkpts0_f_origin": torch.cat(
                    [mkpts0_f_det_origin[valid_mask], add_mkpts0_f_origin]
                ),
                "mkpts1_f_origin": torch.cat(
                    [mkpts1_f_det_origin[valid_mask], add_mkpts1_f_origin]
                ),
            }
        else:
            sub_pixel_matches = {
                "m_bids": b_ids_c[b_ids_det[valid_mask]],
                "mkpts0_f1": mkpts0_f1_det[valid_mask],
                "mkpts1_f1": mkpts1_f1_det[valid_mask],
                "mkpts0_f": mkpts0_f_det[valid_mask],
                "mkpts1_f": mkpts1_f_det[valid_mask],
                "mconf_f": mconf_det,
                "diff": diff[b_ids_det[valid_mask]],
                "diff_points": diff_points,
                "diff_points_conf": diff_points_conf,
                "mkpts0_f1_fine": mkpts0_c_scaled_to_f_det[b_ids_det][valid_mask],
                "mkpts1_f1_fine": mkpts1_c_scaled_to_f_det[b_ids_det][valid_mask],
                "mkpts0_f1_window": mkpts0_f_window_det[valid_mask],
                "mkpts1_f1_window": mkpts1_f_window_det[valid_mask],
                "mkpts0_subref": sub_ref0_det[valid_mask],
                "mkpts1_subref": sub_ref1_det[valid_mask],
                "mkpts0_f1_origin": mkpts0_f1_det_origin[valid_mask],
                "mkpts1_f1_origin": mkpts1_f1_det_origin[valid_mask],
                "mkpts0_f_origin": mkpts0_f_det_origin[valid_mask],
                "mkpts1_f_origin": mkpts1_f_det_origin[valid_mask],
            }

        return sub_pixel_matches
