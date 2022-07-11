import copy
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.events import get_event_storage
from core.utils.solver_utils import build_optimizer_with_params

from core.utils.my_checkpoint import load_timm_pretrained
from mmcv.runner import load_checkpoint

from .net_factory import PCLNETS
from .model_utils import (
    compute_mean_re_te,
    get_rot_mat,
    get_rot_head,
    get_ts_head,
)

from ..losses.l2_loss import L2Loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss

from .pose_scale_from_delta_init import pose_scale_from_delta_init

logger = logging.getLogger(__name__)


class CATRE_disR_shared(nn.Module):
    def __init__(self, cfg, pcl_net, rot_head, ts_head):
        super().__init__()
        assert cfg.MODEL.CATRE.NAME == "CATRE_disR_shared", cfg.MODEL.CATRE.NAME
        self.cfg = cfg
        self.pcl_net = pcl_net
        self.rot_head = rot_head
        self.ts_head = ts_head

    def forward(
        self,
        x,
        tfd_kps,
        init_pose,
        init_scale,
        K_zoom=None,
        obj_class=None,
        gt_ego_rot=None,
        gt_trans=None,
        gt_scale=None,
        obj_kps=None,
        mean_scales=None,
        sym_info=None,
        do_loss=False,
        cur_iter=0,
    ):
        cfg = self.cfg
        net_cfg = cfg.MODEL.CATRE
        rot_head_cfg = net_cfg.ROT_HEAD
        ts_head_cfg = net_cfg.TS_HEAD

        num_classes = net_cfg.NUM_CLASSES
        device = x.device
        bs = x.shape[0]

        pcl_feat = self.pcl_net(x)  # [bs, c, num_p]
        kps_feat = self.pcl_net(tfd_kps)  # [bs, c, num_p]

        flat_pcl_feat = torch.max(pcl_feat, 2, keepdim=False)[0]

        if ts_head_cfg.WITH_KPS_FEATURE:
            flat_kps_feat = torch.max(kps_feat, 2, keepdim=False)[0]
            ts_feat = torch.cat((flat_pcl_feat, flat_kps_feat), dim=1)  # bs, 2c
        else:
            ts_feat = flat_pcl_feat

        # use scale as explicit input
        if ts_head_cfg.WITH_INIT_SCALE:
            ts_feat = torch.cat((ts_feat, init_scale), dim=1)  # [bs, c'+3]
        if ts_head_cfg.WITH_INIT_TRANS:
            init_trans = init_pose[:, :3, 3]
            ts_feat = torch.cat((ts_feat, init_trans), dim=1)  # [bs, c'+3]

        trans_deltas_, scale_deltas_ = self.ts_head(ts_feat)

        rot_feat = torch.cat((pcl_feat, kps_feat), dim=2)  # bs, c, num_pcl+num_kps

        rot_deltas_ = self.rot_head(rot_feat)

        if rot_head_cfg.CLASS_AWARE:
            assert obj_class is not None
            rot_deltas_ = rot_deltas_.view(bs, num_classes, self.pose_head.rot_dim)
            rot_deltas_ = rot_deltas_[torch.arange(bs).to(device), obj_class]
            trans_deltas_ = trans_deltas_.view(bs, num_classes, 3)
            trans_deltas_ = trans_deltas_[torch.arange(bs).to(device), obj_class]

        # convert pred_rot to rot mat -------------------------
        rot_m_deltas = get_rot_mat(rot_deltas_, rot_type=rot_head_cfg.ROT_TYPE)
        # rot_m_deltas, trans_deltas, init_pose --> ego pose -----------------------------
        pred_ego_rot, pred_trans, pred_scale = pose_scale_from_delta_init(
            rot_deltas=rot_m_deltas,
            trans_deltas=trans_deltas_,
            scale_deltas=scale_deltas_,
            rot_inits=init_pose[:, :3, :3],
            trans_inits=init_pose[:, :3, 3],
            scale_inits=init_scale if "iter" in rot_head_cfg.SCLAE_TYPE else mean_scales,
            Ks=K_zoom,  # Ks without zoom # no need
            K_aware=rot_head_cfg.T_TRANSFORM_K_AWARE,
            delta_T_space=rot_head_cfg.DELTA_T_SPACE,
            delta_T_weight=rot_head_cfg.DELTA_T_WEIGHT,
            delta_z_style=rot_head_cfg.DELTA_Z_STYLE,
            eps=1e-4,
            is_allo="allo" in rot_head_cfg.ROT_TYPE,
            scale_type=rot_head_cfg.SCLAE_TYPE,
        )
        pred_pose = torch.cat([pred_ego_rot, pred_trans.view(-1, 3, 1)], dim=-1)

        # NOTE: an ablation setting
        if not cfg.MODEL.REFINE_SCLAE:
            pred_scale = init_scale

        out_dict = {f"pose_{cur_iter}": pred_pose, f"scale_{cur_iter}": pred_scale}
        if not do_loss:  # test
            return out_dict
        else:
            assert gt_ego_rot is not None and (gt_trans is not None)
            mean_re, mean_te = compute_mean_re_te(pred_trans, pred_ego_rot, gt_trans, gt_ego_rot)
            # yapf: disable
            vis_dict = {
                f"vis/error_R_{cur_iter}": mean_re,  # deg
                f"vis/error_t_{cur_iter}": mean_te * 100,  # cm
                f"vis/error_tx_{cur_iter}": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                f"vis/error_ty_{cur_iter}": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                f"vis/error_tz_{cur_iter}": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
                f"vis/tx_pred_{cur_iter}": pred_trans[0, 0].detach().item(),
                f"vis/ty_pred_{cur_iter}": pred_trans[0, 1].detach().item(),
                f"vis/tz_pred_{cur_iter}": pred_trans[0, 2].detach().item(),
                f"vis/tx_delta_{cur_iter}": trans_deltas_[0, 0].detach().item(),
                f"vis/ty_delta_{cur_iter}": trans_deltas_[0, 1].detach().item(),
                f"vis/tz_delta_{cur_iter}": trans_deltas_[0, 2].detach().item(),
                f"vis/tx_gt_{cur_iter}": gt_trans[0, 0].detach().item(),
                f"vis/ty_gt_{cur_iter}": gt_trans[0, 1].detach().item(),
                f"vis/tz_gt_{cur_iter}": gt_trans[0, 2].detach().item(),
            }

            loss_dict = self.catre_loss(
                out_rot=pred_ego_rot, gt_rot=gt_ego_rot,
                out_trans=pred_trans, gt_trans=gt_trans,
                out_scale=pred_scale, gt_scale=gt_scale,
                obj_kps=obj_kps, sym_info=sym_info,
            )

            if net_cfg.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}_{cur_iter}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            # yapf: enable
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict

    def catre_loss(
        self,
        out_rot,
        out_trans,
        out_scale,
        gt_rot=None,
        gt_trans=None,
        gt_scale=None,
        obj_kps=None,
        sym_info=None,
    ):
        cfg = self.cfg
        net_cfg = cfg.MODEL.CATRE
        loss_cfg = net_cfg.LOSS_CFG

        loss_dict = {}
        # point matching loss ---------------
        if loss_cfg.PM_LW > 0:
            assert (obj_kps is not None) and (gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=loss_cfg.PM_LOSS_TYPE,
                beta=loss_cfg.PM_SMOOTH_L1_BETA,
                reduction="mean",
                loss_weight=loss_cfg.PM_LW,
                symmetric=loss_cfg.PM_LOSS_SYM,
                disentangle_t=loss_cfg.PM_DISENTANGLE_T,
                disentangle_z=loss_cfg.PM_DISENTANGLE_Z,
                t_loss_use_points=loss_cfg.PM_T_USE_POINTS,
                r_only=loss_cfg.PM_R_ONLY,
                with_scale=loss_cfg.PM_WITH_SCALE,
                use_bbox=loss_cfg.PM_USE_BBOX,
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=obj_kps,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                pred_scales=out_scale,
                gt_scales=gt_scale,
                sym_infos=sym_info,
            )
            loss_dict.update(loss_pm_dict)

        # rot_loss (symmetry-aware) ----------
        if loss_cfg.ROT_LW > 0:
            # NOTE: for now all sym_infos are about y axis. If new sym_type is introduced, please change the code here.
            sym_mask = torch.tensor([0 if sym is None else 1 for sym in sym_info]).to(out_rot.device)
            out_rot_nosym = torch.index_select(out_rot, dim=0, index=torch.where(sym_mask == 0)[0])
            gt_rot_nosym = torch.index_select(gt_rot, dim=0, index=torch.where(sym_mask == 0)[0])
            out_rot_sym = torch.index_select(out_rot, dim=0, index=torch.where(sym_mask == 1)[0])
            gt_rot_sym = torch.index_select(gt_rot, dim=0, index=torch.where(sym_mask == 1)[0])

            # for non-sym object
            if out_rot_nosym.shape[0] > 0:
                if loss_cfg.ROT_LOSS_TYPE == "angular":
                    loss_dict["loss_rot"] = angular_distance(out_rot_nosym, gt_rot_nosym)
                elif loss_cfg.ROT_LOSS_TYPE == "L2":
                    loss_dict["loss_rot"] = rot_l2_loss(out_rot_nosym, gt_rot_nosym)
                else:
                    raise ValueError(f"Unknown rot loss type: {loss_cfg.ROT_LOSS_TYPE}")
                loss_dict["loss_rot"] *= loss_cfg.ROT_LW

            # for sym object, just the second column
            if out_rot_sym.shape[0] > 0:
                if loss_cfg.ROT_YAXIS_LOSS_TYPE == "L1":
                    loss_dict["loss_yaxis_rot"] = nn.L1Loss(reduction="mean")(out_rot_sym[:, :, 1], gt_rot_sym[:, :, 1])
                elif loss_cfg.ROT_YAXIS_LOSS_TYPE == "smoothL1":
                    loss_dict["loss_yaxis_rot"] = nn.SmoothL1Loss(reduction="mean")(
                        out_rot_sym[:, :, 1], gt_rot_sym[:, :, 1]
                    )
                elif loss_cfg.ROT_YAXIS_LOSS_TYPE == "L2":
                    loss_dict["loss_yaxis_rot"] = L2Loss(reduction="mean")(out_rot_sym[:, :, 1], gt_rot_sym[:, :, 1])
                elif loss_cfg.ROT_YAXIS_LOSS_TYPE == "angular":
                    loss_dict["loss_yaxis_rot"] = angular_distance(out_rot_sym[:, :, 1], gt_rot_sym[:, :, 1])
                else:
                    raise ValueError(f"Unknown rot yaxis loss type: {loss_cfg.ROT_YAXIS_LOSS_TYPE}")
                loss_dict["loss_yaxis_rot"] *= loss_cfg.ROT_LW

        # trans loss ------------------
        if loss_cfg.TRANS_LW > 0:
            if loss_cfg.TRANS_LOSS_DISENTANGLE:
                # NOTE: disentangle xy/z
                if loss_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_xy"] = nn.L1Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_xy"] = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_xy"] = nn.MSELoss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_xy"] *= loss_cfg.TRANS_LW
                loss_dict["loss_trans_z"] *= loss_cfg.TRANS_LW
            else:
                if loss_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_LPnP"] = nn.L1Loss(reduction="mean")(out_trans, gt_trans)
                elif loss_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_LPnP"] = L2Loss(reduction="mean")(out_trans, gt_trans)
                elif loss_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_LPnP"] = nn.MSELoss(reduction="mean")(out_trans, gt_trans)
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_LPnP"] *= loss_cfg.TRANS_LW

        # scale loss ---------------------
        if loss_cfg.SCALE_LW > 0:
            assert cfg.MODEL.REFINE_SCLAE
            if loss_cfg.SCALE_LOSS_TYPE == "L1":
                loss_dict["loss_scale"] = nn.L1Loss(reduction="mean")(out_scale, gt_scale)
            elif loss_cfg.SCALE_LOSS_TYPE == "L2":
                loss_dict["loss_scale"] = L2Loss(reduction="mean")(out_scale, gt_scale)
            elif loss_cfg.SCALE_LOSS_TYPE == "MSE":
                loss_dict["loss_scale"] = nn.MSELoss(reduction="mean")(out_scale, gt_scale)
            else:
                raise ValueError(f"Unknown scale loss type: {loss_cfg.SCALE_LOSS_TYPE}")
            loss_dict["loss_scale"] *= loss_cfg.SCALE_LW

        return loss_dict


def build_model_optimizer(cfg, is_test=False):
    pcl_net_cfg = cfg.MODEL.CATRE.PCLNET

    params_lr_list = []

    # pcl_net ----------------------------------------------------------
    init_pcl_net_args = copy.deepcopy(pcl_net_cfg.INIT_CFG)
    pcl_net_type = init_pcl_net_args.pop("type")

    pcl_net = PCLNETS[pcl_net_type](**init_pcl_net_args)
    if pcl_net_cfg.FREEZE:
        for param in pcl_net_cfg.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {"params": filter(lambda p: p.requires_grad, pcl_net.parameters()), "lr": float(cfg.SOLVER.BASE_LR)}
        )

    # disentangle pose head -----------------------------------------------------
    rot_head, rot_head_params = get_rot_head(cfg)
    params_lr_list.extend(rot_head_params)

    ts_head, ts_head_params = get_ts_head(cfg)
    params_lr_list.extend(ts_head_params)

    # ================================================
    # build model
    model = CATRE_disR_shared(cfg, pcl_net, rot_head, ts_head)

    # get optimizer
    if is_test:
        optimizer = None
    else:
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## pcl_net initialization
        pcl_net_pretrained = pcl_net_cfg.get("PRETRAINED", "")
        if pcl_net_pretrained == "":
            logger.warning("Randomly initialize weights for pcl_net!")
        elif pcl_net_pretrained in ["timm", "internal"]:
            # skip if it has already been initialized by pretrained=True
            logger.info("Check if the pcl_net has been initialized with its own method!")
            if pcl_net_pretrained == "timm":
                if init_pcl_net_args.pretrained and init_pcl_net_args.in_chans != 3:
                    load_timm_pretrained(
                        model.pcl_net, in_chans=init_pcl_net_args.in_chans, adapt_input_mode="custom", strict=False
                    )
                    logger.warning("override input conv weight adaptation of timm")
        else:
            # initialize pcl_net with official weights
            tic = time.time()
            logger.info(f"load pcl_net weights from: {pcl_net_pretrained}")
            load_checkpoint(model.pcl_net, pcl_net_pretrained, strict=False, logger=logger)
            logger.info(f"load pcl_net weights took: {time.time() - tic}s")

    model.to(torch.device(cfg.MODEL.DEVICE))

    return model, optimizer
