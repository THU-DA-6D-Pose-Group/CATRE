import torch
import copy
import numpy as np
from lib.pysixd.pose_error import re, te
from core.utils.pose_utils import quat2mat_torch
from core.utils.rot_reps import rot6d_to_mat_batch
from core.utils import lie_algebra, quaternion_lf
from .net_factory import HEADS, PCLNETS


def get_rot_dim(rot_type):
    if rot_type in ["allo_quat", "ego_quat"]:
        rot_dim = 4
    elif rot_type in [
        "allo_log_quat",
        "ego_log_quat",
        "allo_lie_vec",
        "ego_lie_vec",
    ]:
        rot_dim = 3
    elif rot_type in ["allo_rot6d", "ego_rot6d"]:
        rot_dim = 6
    else:
        raise ValueError(f"Unknown rot_type: {rot_type}")
    return rot_dim


def get_rot_mat(rot, rot_type):
    if rot_type in ["ego_quat", "allo_quat"]:
        rot_m = quat2mat_torch(rot)
    elif rot_type in ["ego_log_quat", "allo_log_quat"]:
        # from latentfusion (lf)
        rot_m = quat2mat_torch(quaternion_lf.qexp(rot))
    elif rot_type in ["ego_lie_vec", "allo_lie_vec"]:
        rot_m = lie_algebra.lie_vec_to_rot(rot)
    elif rot_type in ["ego_rot6d", "allo_rot6d"]:
        rot_m = rot6d_to_mat_batch(rot)
    else:
        raise ValueError(f"Wrong pred_rot type: {rot_type}")
    return rot_m


def get_kps_net(cfg):
    net_cfg = cfg.MODEL.CATRE
    kps_net_cfg = net_cfg.KPSNET
    params_lr_list = []

    kps_net_init_cfg = copy.deepcopy(kps_net_cfg.INIT_CFG)
    kps_net_type = kps_net_init_cfg.pop("type")

    kps_net = PCLNETS[kps_net_type](**kps_net_init_cfg)
    if kps_net_cfg.FREEZE:
        for param in kps_net.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, kps_net.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * kps_net_cfg.LR_MULT,
            }
        )
    return kps_net, params_lr_list


def get_rot_head(cfg):
    net_cfg = cfg.MODEL.CATRE
    rot_head_cfg = net_cfg.ROT_HEAD
    params_lr_list = []

    rot_num_classes = net_cfg.NUM_CLASSES if rot_head_cfg.CLASS_AWARE else 1

    rot_head_init_cfg = copy.deepcopy(rot_head_cfg.INIT_CFG)
    rot_head_type = rot_head_init_cfg.pop("type")

    rot_head_init_cfg.update(num_classes=rot_num_classes)
    rot_head = HEADS[rot_head_type](**rot_head_init_cfg)
    if rot_head_cfg.FREEZE:
        for param in rot_head.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, rot_head.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * rot_head_cfg.LR_MULT,
            }
        )
    return rot_head, params_lr_list


def get_t_head(cfg):
    net_cfg = cfg.MODEL.CATRE
    t_head_cfg = net_cfg.T_HEAD
    params_lr_list = []

    num_classes = net_cfg.NUM_CLASSES if net_cfg.ROT_HEAD.CLASS_AWARE else 1

    t_head_init_cfg = copy.deepcopy(t_head_cfg.INIT_CFG)
    t_head_type = t_head_init_cfg.pop("type")

    t_head_init_cfg.update(num_classes=num_classes)
    t_head = HEADS[t_head_type](**t_head_init_cfg)
    if t_head_cfg.FREEZE:
        for param in t_head.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, t_head.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * t_head_cfg.LR_MULT,
            }
        )
    return t_head, params_lr_list


def get_s_head(cfg):
    net_cfg = cfg.MODEL.CATRE
    s_head_cfg = net_cfg.S_HEAD
    params_lr_list = []

    num_classes = net_cfg.NUM_CLASSES if net_cfg.ROT_HEAD.CLASS_AWARE else 1

    s_head_init_cfg = copy.deepcopy(s_head_cfg.INIT_CFG)
    s_head_type = s_head_init_cfg.pop("type")

    s_head_init_cfg.update(num_classes=num_classes)
    s_head = HEADS[s_head_type](**s_head_init_cfg)
    if s_head_cfg.FREEZE:
        for param in s_head.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, s_head.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * s_head_cfg.LR_MULT,
            }
        )
    return s_head, params_lr_list


def get_ts_head(cfg):
    net_cfg = cfg.MODEL.CATRE
    ts_head_cfg = net_cfg.TS_HEAD
    params_lr_list = []

    num_classes = net_cfg.NUM_CLASSES if net_cfg.ROT_HEAD.CLASS_AWARE else 1

    ts_head_init_cfg = copy.deepcopy(ts_head_cfg.INIT_CFG)
    ts_head_type = ts_head_init_cfg.pop("type")

    ts_head_init_cfg.update(num_classes=num_classes)
    ts_head = HEADS[ts_head_type](**ts_head_init_cfg)
    if ts_head_cfg.FREEZE:
        for param in ts_head.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, ts_head.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * ts_head_cfg.LR_MULT,
            }
        )
    return ts_head, params_lr_list


def get_nocs_head(cfg):
    net_cfg = cfg.MODEL.CATRE
    nocs_head_cfg = net_cfg.NOCS_HEAD
    params_lr_list = []

    num_classes = net_cfg.NUM_CLASSES if net_cfg.ROT_HEAD.CLASS_AWARE else 1

    nocs_head_init_cfg = copy.deepcopy(nocs_head_cfg.INIT_CFG)
    nocs_head_type = nocs_head_init_cfg.pop("type")

    nocs_head_init_cfg.update(num_classes=num_classes)

    nocs_head = HEADS[nocs_head_type](**nocs_head_init_cfg)

    if nocs_head_cfg.FREEZE:
        for param in nocs_head.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, nocs_head.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * nocs_head_cfg.LR_MULT,
            }
        )
    return nocs_head, params_lr_list


def get_pose_head(cfg):
    net_cfg = cfg.MODEL.CATRE
    pose_head_cfg = net_cfg.POSE_HEAD
    params_lr_list = []
    rot_type = pose_head_cfg.ROT_TYPE

    rot_dim = get_rot_dim(rot_type)
    pose_num_classes = net_cfg.NUM_CLASSES if pose_head_cfg.CLASS_AWARE else 1

    pose_head_init_cfg = copy.deepcopy(pose_head_cfg.INIT_CFG)
    pose_head_type = pose_head_init_cfg.pop("type")

    pose_head_init_cfg.update(rot_dim=rot_dim, num_classes=pose_num_classes)
    pose_head = HEADS[pose_head_type](**pose_head_init_cfg)
    if pose_head_cfg.FREEZE:
        for param in pose_head.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, pose_head.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * pose_head_cfg.LR_MULT,
            }
        )
    return pose_head, params_lr_list


def compute_mean_re_te(pred_transes, pred_rots, gt_transes, gt_rots):
    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    gt_transes = gt_transes.detach().cpu().numpy()
    gt_rots = gt_rots.detach().cpu().numpy()

    bs = pred_rots.shape[0]
    R_errs = np.zeros((bs,), dtype=np.float32)
    T_errs = np.zeros((bs,), dtype=np.float32)
    for i in range(bs):
        R_errs[i] = re(pred_rots[i], gt_rots[i])
        T_errs[i] = te(pred_transes[i], gt_transes[i])
    return R_errs.mean(), T_errs.mean()
