import sys
import os.path as osp
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.pose_utils import quat2mat_torch
from .l2_loss import L2Loss
from fvcore.nn import smooth_l1_loss
from lib.utils.utils import dprint
from core.utils.pose_utils import get_closest_rot_batch
from core.catre.engine.engine_utils import get_normed_bbox
import logging
from detectron2.utils.logger import log_first_n
from lib.pysixd.misc import transform_pts_batch, transform_normed_pts_batch


logger = logging.getLogger(__name__)


class PyPMLoss(nn.Module):
    """Point matching loss."""

    def __init__(
        self,
        loss_type="l1",
        beta=1.0,
        reduction="mean",
        loss_weight=1.0,
        disentangle_t=False,
        disentangle_z=False,
        t_loss_use_points=False,
        symmetric=False,
        r_only=False,
        with_scale=False,
        use_bbox=False,
    ):
        """
        Args:
            loss_type:
            beta:
            reduction:
            loss_weight:
            norm_by_extent:
            disentangle_t:  disentangle R/T
            disentangle_z:  disentangle R/xy/z
            t_loss_use_points: only used for disentangled loss; whether to
                use points to compute losses of T
            symmetric:
            r_only:
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.disentangle_t = disentangle_t
        self.disentangle_z = disentangle_z
        if disentangle_z and (disentangle_t is False):
            log_first_n(logging.WARNING, "disentangle_z means: disentangle R/xy/z", n=1)
            self.disentangle_t = True

        if (disentangle_t is False) and (disentangle_z is False):
            # if not disentangled, must use points to compute t loss
            self.t_loss_use_points = True
        else:
            self.t_loss_use_points = t_loss_use_points
        self.symmetric = symmetric
        self.r_only = r_only
        self.with_scale = with_scale
        self.use_bbox = use_bbox

        self.loss_type = loss_type = loss_type.lower()
        if loss_type == "smooth_l1":
            self.loss_func = partial(smooth_l1_loss, beta=beta, reduction=reduction)
        elif loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction=reduction)
        elif loss_type == "mse":
            self.loss_func = nn.MSELoss(reduction=reduction)  # squared L2
        elif loss_type == "l2":
            self.loss_func = L2Loss(reduction=reduction)
        else:
            raise ValueError("loss type {} not supported.".format(loss_type))

    def forward(
        self,
        pred_rots,
        gt_rots,
        points,
        pred_transes=None,
        gt_transes=None,
        pred_scales=None,
        gt_scales=None,
        sym_infos=None,
    ):
        """
        pred_rots: [B, 3, 3]
        gt_rots: [B, 3, 3] or [B, 4]
        points: [B, n, 3]

        pred_transes: [B, 3]
        gt_transes: [B, 3]
        extents: [B, 3]
        sym_infos: list [Kx3x3 or None],
            stores K rotations regarding symmetries, if not symmetric, None
        """
        if gt_rots.shape[-1] == 4:
            gt_rots = quat2mat_torch(gt_rots)

        if self.symmetric:
            assert sym_infos is not None
            gt_rots = get_closest_rot_batch(pred_rots, gt_rots, sym_infos=sym_infos)

        if self.use_bbox:
            bs = pred_scales.shape[0]
            to_float_args = {"dtype": torch.float32, "device": "cuda", "non_blocking": True}
            points = get_normed_bbox(bs).to(**to_float_args)

        # [B, n, 3]
        if self.with_scale:
            points_est = transform_normed_pts_batch(points, pred_rots, t=None, scale=pred_scales)
            points_tgt = transform_normed_pts_batch(points, gt_rots, t=None, scale=gt_scales)
        else:
            points_est = transform_normed_pts_batch(points, pred_rots, t=None, scale=None)
            points_tgt = transform_normed_pts_batch(points, gt_rots, t=None, scale=None)

        if self.r_only:
            loss = self.loss_func(points_est, points_tgt)
            loss_dict = {"loss_PM_R": 3 * loss * self.loss_weight}
        else:
            assert pred_transes is not None and gt_transes is not None, "pred_transes and gt_transes should be given"

            if self.disentangle_z:  # R/xy/z
                if self.t_loss_use_points:
                    points_tgt_RT = points_tgt + gt_transes.view(-1, 1, 3)
                    # using gt T
                    points_est_R = points_est + gt_transes.view(-1, 1, 3)

                    # using gt R,z
                    pred_transes_xy = pred_transes.clone()
                    pred_transes_xy[:, 2] = gt_transes[:, 2]
                    points_est_xy = points_tgt + pred_transes_xy.view(-1, 1, 3)

                    # using gt R/xy
                    pred_transes_z = pred_transes.clone()
                    pred_transes_z[:, :2] = gt_transes[:, :2]
                    points_est_z = points_tgt + pred_transes_z.view(-1, 1, 3)

                    loss_R = self.loss_func(points_est_R, points_tgt_RT)
                    loss_xy = self.loss_func(points_est_xy, points_tgt_RT)
                    loss_z = self.loss_func(points_est_z, points_tgt_RT)
                    loss_dict = {
                        "loss_PM_R": 3 * loss_R * self.loss_weight,
                        "loss_PM_xy": 3 * loss_xy * self.loss_weight,
                        "loss_PM_z": 3 * loss_z * self.loss_weight,
                    }
                else:
                    loss_R = self.loss_func(points_est, points_tgt)
                    loss_xy = self.loss_func(pred_transes[:, :2], gt_transes[:, :2])
                    loss_z = self.loss_func(pred_transes[:, 2], gt_transes[:, 2])
                    loss_dict = {
                        "loss_PM_R": 3 * loss_R * self.loss_weight,
                        "loss_PM_xy_noP": loss_xy,
                        "loss_PM_z_noP": loss_z,
                    }
            elif self.disentangle_t:  # R/t
                if self.t_loss_use_points:
                    points_tgt_RT = points_tgt + gt_transes.view(-1, 1, 3)
                    # using gt T
                    points_est_R = points_est + gt_transes.view(-1, 1, 3)

                    # using gt R
                    points_est_T = points_tgt + pred_transes.view(-1, 1, 3)

                    loss_R = self.loss_func(points_est_R, points_tgt_RT)
                    loss_T = self.loss_func(points_est_T, points_tgt_RT)
                    loss_dict = {
                        "loss_PM_R": 3 * loss_R * self.loss_weight,
                        "loss_PM_T": 3 * loss_T * self.loss_weight,
                    }
                else:
                    loss_R = self.loss_func(points_est, points_tgt)
                    loss_T = self.loss_func(pred_transes, gt_transes)
                    loss_dict = {
                        "loss_PM_R": 3 * loss_R * self.loss_weight,
                        "loss_PM_T_noP": loss_T,
                    }
            else:  # no disentangle
                points_tgt_RT = points_tgt + gt_transes.view(-1, 1, 3)
                points_est_RT = points_est + pred_transes.view(-1, 1, 3)
                loss = self.loss_func(points_est_RT, points_tgt_RT)
                loss_dict = {"loss_PM_RT": 3 * loss * self.loss_weight}
        # NOTE: 3 is for mean reduction on the point dim
        return loss_dict
