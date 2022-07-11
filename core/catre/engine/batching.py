import torch
from .engine_utils import get_init_pose_train, get_normed_kps, get_init_scale_train
from .engine_utils import aug_3d_bbox, aug_RT
from .batch_test import batch_data_test, batch_updater_test
from lib.vis_utils.image import grid_show, heatmap
from lib.pysixd.misc import transform_pts_batch, transform_normed_pts_batch


def batch_data(cfg, data, device="cuda", phase="train", dtype=torch.float32):
    if phase == "test":
        return batch_data_test(cfg, data, device=device, dtype=dtype)

    input_cfg = cfg.INPUT
    # batch training data and flatten
    tensor_kwargs = {"dtype": dtype, "device": device}
    to_float_args = {"dtype": dtype, "device": device, "non_blocking": True}
    to_long_args = {"dtype": torch.long, "device": device, "non_blocking": True}

    batch = {}
    num_imgs = len(data)

    # construct by flattening instance data =============================
    batch["pcl"] = torch.cat([d["instances"].pcl for d in data], dim=0).to(**to_float_args)

    batch["obj_cls"] = torch.cat([d["instances"].obj_classes for d in data], dim=0).to(**to_long_args)
    batch["obj_pose"] = torch.cat([d["instances"].obj_poses.tensor for d in data], dim=0).to(**to_float_args)
    batch["obj_scale"] = torch.cat([d["instances"].obj_scales for d in data], dim=0).to(**to_float_args)

    # batch["obj_visib_mask"] = torch.cat([d["instances"].obj_visib_masks.tensor for d in data], dim=0).to(
    #     **to_float_args
    # )
    # batch["obj_trunc_mask"] = torch.cat([d["instances"].obj_trunc_masks.tensor for d in data], dim=0).to(
    #     **to_float_args
    # )

    batch["obj_mean_points"] = torch.cat([d["instances"].obj_mean_points for d in data], dim=0).to(**to_float_args)
    batch["obj_mean_scales"] = torch.cat([d["instances"].obj_mean_scales for d in data], dim=0).to(**to_float_args)

    if "last_frame" in cfg.INPUT.INIT_POSE_TYPE_TRAIN:
        batch["last_frame_poses"] = torch.cat([d["instances"].last_frame_poses for d in data], dim=0).to(
            **to_float_args
        )

    if cfg.INPUT.KPS_TYPE.lower() == "fps":
        batch["obj_fps_points"] = torch.cat([d["instances"].obj_fps_points for d in data], dim=0).to(**to_float_args)

    num_insts_per_im = [len(d["instances"]) for d in data]
    num_insts_all = len(batch["obj_cls"])
    K_list = []
    sym_infos_list = []
    im_ids = []
    inst_ids = []  # the idx in the current img, should be used with im_ids
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

    # keep max_objs ----------------------------
    n_obj = min(cfg.DATALOADER.MAX_OBJS_TRAIN, num_insts_all)
    for _k in batch:
        if len(batch[_k]) == num_insts_all:
            batch[_k] = batch[_k][:n_obj]

    if input_cfg.WITH_IMG:
        batch["img"] = torch.stack([d["image"] for d in data]).to(**to_float_args)

    if input_cfg.WITH_DEPTH:
        batch["depth_obs"] = torch.stack([d["depth"] for d in data], dim=0).to(**to_float_args)

    # do some augmentation here --------------------------------------------
    if torch.rand(1) < input_cfg.BBOX3D_AUG_PROB:
        aug_3d_bbox(batch)

    if torch.rand(1) < input_cfg.RT_AUG_PROB:
        aug_RT(batch)

    return batch


def batch_updater(
    cfg,
    batch,
    cur_iter=1,
    poses_est=None,
    scales_est=None,
    device="cuda",
    dtype=torch.float32,
    phase="train",
):
    if phase == "test":
        return batch_updater_test(
            cfg,
            batch,
            poses_est=poses_est,
            scales_est=scales_est,
            device=device,
            dtype=dtype,
        )
    # batch updater for train =========================================
    tensor_kwargs = {"dtype": dtype, "device": device}
    to_float_args = {"dtype": dtype, "device": device, "non_blocking": True}

    n_obj = batch["obj_pose"].shape[0]
    if poses_est is None:
        # init pose --------------------------------------------------------------------------
        get_init_pose_train(cfg, batch, **tensor_kwargs)  # obj_pose_est
    else:
        batch["obj_pose_est"] = poses_est

    if scales_est is None:
        if cfg.MODEL.REFINE_SCLAE:
            get_init_scale_train(cfg, batch, **tensor_kwargs)  # obj_scale_est
        else:
            batch["obj_scale_est"] = batch["obj_scale"].detach().clone()
    else:
        batch["obj_scale_est"] = scales_est

    if "obj_kps" not in batch:
        get_normed_kps(cfg, batch, **to_float_args)  # obj_kps

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

    # done batch update train------------------------------------------
