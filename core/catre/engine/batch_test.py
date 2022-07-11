import os.path as osp
import random
import torch

from .engine_utils import get_normed_kps
from lib.vis_utils.image import heatmap, grid_show
from lib.pysixd.misc import transform_normed_pts_batch


def batch_data_test(cfg, data, device="cuda", dtype=torch.float32):
    # batch test data and flatten
    tensor_kwargs = {"dtype": dtype, "device": device}
    to_float_args = {"dtype": dtype, "device": device, "non_blocking": True}
    to_long_args = {"dtype": torch.long, "device": device, "non_blocking": True}

    batch = {}
    num_imgs = len(data)
    # construct flattened instance data =============================
    batch["obj_cls"] = torch.cat([d["instances"].obj_classes for d in data], dim=0).to(**to_long_args)
    batch["obj_bbox"] = torch.cat([d["instances"].obj_boxes.tensor for d in data], dim=0).to(**to_float_args)
    # NOTE: initial pose or the output pose estimate
    batch["obj_pose_est"] = torch.cat([d["instances"].obj_poses.tensor for d in data], dim=0).to(**to_float_args)
    batch["obj_scale_est"] = torch.cat([d["instances"].obj_scales for d in data], dim=0).to(**to_float_args)

    batch["obj_mean_points"] = torch.cat([d["instances"].obj_mean_points for d in data], dim=0).to(**to_float_args)
    batch["obj_mean_scales"] = torch.cat([d["instances"].obj_mean_scales for d in data], dim=0).to(**to_float_args)

    if cfg.INPUT.KPS_TYPE.lower() == "fps":
        # NOTE: only an ablation setting!
        batch["obj_fps_points"] = torch.cat([d["instances"].obj_fps_points for d in data], dim=0).to(**to_float_args)

    num_insts_per_im = [len(d["instances"]) for d in data]
    n_obj = len(batch["obj_cls"])
    K_list = []
    sym_infos_list = []
    im_ids = []
    inst_ids = []
    for i_im in range(num_imgs):
        sym_infos_list.extend(data[i_im]["instances"].obj_sym_infos)
        for i_inst in range(num_insts_per_im[i_im]):
            im_ids.append(i_im)
            inst_ids.append(i_inst)
            K_list.append(data[i_im]["cam"].clone())

    batch["im_id"] = torch.tensor(im_ids, **tensor_kwargs)
    batch["inst_id"] = torch.tensor(inst_ids, **tensor_kwargs)
    batch["K"] = torch.stack(K_list, dim=0).to(**to_float_args)
    batch["sym_info"] = sym_infos_list

    input_cfg = cfg.INPUT

    batch["pcl"] = torch.cat([d["instances"].pcl for d in data], dim=0).to(**to_float_args)

    if input_cfg.WITH_IMG:
        batch["img"] = torch.stack([d["image"] for d in data]).to(**to_float_args)

    if input_cfg.WITH_DEPTH:
        batch["depth_obs"] = torch.stack([d["depth"] for d in data], dim=0).to(**to_float_args)

    return batch


def batch_updater_test(cfg, batch, poses_est=None, scales_est=None, device="cuda", dtype=torch.float32):
    """
    iter=0: poses_est=None, obj_pose_est is from data loader
    if REFINE_SCLAE is False, keep init_scale unchanged from iter 0 ~ max_num
    """
    tensor_kwargs = {"dtype": dtype, "device": device}
    to_float_args = {"dtype": dtype, "device": device, "non_blocking": True}

    n_obj = batch["obj_cls"].shape[0]
    if poses_est is not None:
        batch["obj_pose_est"] = poses_est

    if scales_est is not None and cfg.MODEL.REFINE_SCLAE:
        batch["obj_scale_est"] = scales_est

    if "obj_kps" not in batch:
        get_normed_kps(cfg, batch, **to_float_args)

    r_est = batch["obj_pose_est"][:, :3, :3]
    t_est = batch["obj_pose_est"][:, :3, 3:4]
    s_est = batch["obj_scale_est"]

    tfd_kps = transform_normed_pts_batch(
        batch["obj_kps"],
        r_est,
        t=None if cfg.INPUT.ZERO_CENTER_INPUT else t_est,
        scale=s_est,
    )

    batch["tfd_kps"] = tfd_kps.permute(0, 2, 1)  # [bs, 3, num_k]

    if cfg.INPUT.ZERO_CENTER_INPUT:
        batch["x"] = batch["pcl"].permute(0, 2, 1) - t_est.view(n_obj, 3, 1)  #  [bs, 3, num_k] - [bs, 3, 1]
    else:
        batch["x"] = batch["pcl"].permute(0, 2, 1)

    # done batch update test------------------------------------------
