import os.path as osp
import random
from IPython.core.pylabtools import figsize
from PIL.ImageOps import scale
from mmcv.visualization import color
import torch
import numpy as np
import math
import itertools
from core.utils.pose_aug import aug_poses_normal, aug_scale_normal
from lib.pysixd.transform import random_rotation_matrix
from lib.vis_utils.image import grid_show, heatmap
from lib.pysixd import misc
from core.utils.pose_utils import rot_from_axangle_chain


def get_normed_kps(cfg, batch, **to_float_args):
    kps_type = cfg.INPUT.KPS_TYPE
    if kps_type.lower() == "bbox":
        scale_est = batch["obj_scale_est"]
        bboxes = get_normed_bbox(scale_est.shape[0])
        batch["obj_kps"] = bboxes.to(**to_float_args)
    elif kps_type.lower() == "mean_shape":
        batch["obj_kps"] = batch["obj_mean_points"].clone()
    elif kps_type.lower() == "fps":
        # NOTE: use obj_scale_est here, train: gt_scale, test: init_scale (with noise)
        batch["obj_kps"] = norm_fps_points(batch["obj_fps_points"], batch["obj_scale_est"]).to(**to_float_args)
    elif kps_type.lower() == "axis":
        scale_est = batch["obj_scale_est"]
        num_kps = cfg.INPUT.NUM_KPS
        with_neg = cfg.INPUT.WITH_NEG_AXIS
        axises = get_normed_axis(scale_est.shape[0], num_kps, with_neg)
        batch["obj_kps"] = axises.to(**to_float_args)
    else:
        raise NotImplementedError(f"Unknown keypoints type {kps_type}")


def norm_fps_points(fps_points, scale):
    return fps_points / scale.unsqueeze(1)  # (B, V, 3)


def get_normed_axis(bs, num_kps=4, with_neg=False):
    num_per_axis = (num_kps - 1) // 3
    if with_neg:
        start = -0.5
        l = 1
    else:
        start = 0
        l = 0.5
    x_points = torch.tensor([[start + l * i / num_per_axis, 0, 0] for i in range(1, num_per_axis + 1)])
    y_points = torch.tensor([[0, start + l * i / num_per_axis, 0] for i in range(1, num_per_axis + 1)])
    z_points = torch.tensor([[0, 0, start + l * i / num_per_axis] for i in range(1, num_per_axis + 1)])
    axis = torch.cat(
        (
            x_points,
            y_points,
            z_points,
            torch.tensor([[0, 0, 0]]),  # with origin
        ),
        dim=0,
    )
    axises = torch.stack([axis for i in range(bs)], dim=0)
    return axises


def get_normed_bbox(bs):
    bbox = torch.tensor(
        [
            [1 / 2, 1 / 2, 1 / 2],
            [-1 / 2, 1 / 2, 1 / 2],
            [-1 / 2, -1 / 2, 1 / 2],
            [1 / 2, -1 / 2, 1 / 2],
            [1 / 2, 1 / 2, -1 / 2],
            [-1 / 2, 1 / 2, -1 / 2],
            [-1 / 2, -1 / 2, -1 / 2],
            [1 / 2, -1 / 2, -1 / 2],
        ]
    )
    bboxes = torch.stack([bbox for i in range(bs)], dim=0)
    return bboxes


def get_bbox_from_scale_batch(scales):
    """scale shape (B, 3)"""

    minx, maxx = -scales[:, 0] / 2, scales[:, 0] / 2
    miny, maxy = -scales[:, 1] / 2, scales[:, 1] / 2
    minz, maxz = -scales[:, 2] / 2, scales[:, 2] / 2

    bboxes = torch.stack(
        (
            torch.stack((maxx, maxy, maxz), dim=1),
            torch.stack((minx, maxy, maxz), dim=1),
            torch.stack((minx, miny, maxz), dim=1),
            torch.stack((maxx, miny, maxz), dim=1),
            torch.stack((maxx, maxy, minz), dim=1),
            torch.stack((minx, maxy, minz), dim=1),
            torch.stack((minx, miny, minz), dim=1),
            torch.stack((maxx, miny, minz), dim=1),
        ),
        dim=1,
    )  # bs, num_k, 3

    return bboxes


def aug_3d_bbox(
    batch, shift_sx=(0.8, 1.2), shift_sy=(0.8, 1.2), shift_sz=(0.8, 1.2), device="cuda", dtype=torch.float32
):
    # generate aug parameters
    ex, ey, ez = torch.rand(3)
    ex = ex * (shift_sx[1] - shift_sx[0]) + shift_sx[0]
    ey = ey * (shift_sy[1] - shift_sy[0]) + shift_sy[0]
    ez = ez * (shift_sz[1] - shift_sz[0]) + shift_sz[0]

    tensor_kwargs = {"dtype": dtype, "device": device}
    to_float_args = {"dtype": dtype, "device": device, "non_blocking": True}
    pcls_aug = []
    scales_aug = []
    for pcl, pose, scale, sym_info in zip(batch["pcl"], batch["obj_pose"], batch["obj_scale"], batch["sym_info"]):
        R = pose[:, :3]
        t = pose[:, 3]
        pcl_reproj = torch.mm(R.T, (pcl - t.view(1, 3)).T).T

        if sym_info is not None:  # y axis symmetry
            exz = (ex + ez) / 2
            ratios = torch.tensor((exz, ey, exz)).to(**tensor_kwargs)
        else:
            ratios = torch.tensor((ex, ey, ez)).to(**tensor_kwargs)

        pcl_reproj = pcl_reproj * ratios.unsqueeze(0)  # (P, 3) * (1, 3)
        scale_aug = scale * ratios
        pcl_aug = torch.mm(R, pcl_reproj.T) + t.view(3, 1)

        scales_aug.append(scale_aug)
        pcls_aug.append(pcl_aug.T)

    batch["obj_scale"] = torch.stack(scales_aug).contiguous().to(**to_float_args)
    batch["pcl"] = torch.stack(pcls_aug).contiguous().to(**to_float_args)


def aug_RT(batch, shift_tx=0.005, shift_ty=0.005, shift_tz=0.025, shift_rot=15.0, device="cuda", dtype=torch.float32):
    tensor_kwargs = {"dtype": dtype, "device": device}
    to_float_args = {"dtype": dtype, "device": device, "non_blocking": True}

    # generate aug parameters
    rx, ry, rz = torch.rand(3) * shift_rot * 2 - shift_rot
    tx = torch.rand(1) * shift_tx * 2 - shift_tx
    ty = torch.rand(1) * shift_ty * 2 - shift_ty
    tz = torch.rand(1) * shift_tz * 2 - shift_tz
    delta_r = get_rotation_torch(rx, ry, rz).to(**tensor_kwargs)
    delta_t = torch.tensor((tx, ty, tz)).to(**tensor_kwargs)

    pcls_aug = []
    Rs_aug = []
    ts_aug = []
    for pcl, pose in zip(batch["pcl"], batch["obj_pose"]):
        R = pose[:, :3]
        t = pose[:, 3]

        pcl_aug = torch.mm(delta_r, (pcl + delta_t.unsqueeze(0)).T).T
        R_aug = torch.mm(delta_r, R)
        t_aug = torch.mm(delta_r, (t + delta_t).view(3, 1))

        pcls_aug.append(pcl_aug)
        Rs_aug.append(R_aug)
        ts_aug.append(t_aug)

    Rs_aug = torch.stack(Rs_aug)
    ts_aug = torch.stack(ts_aug)
    batch["obj_pose"] = torch.cat((Rs_aug, ts_aug), dim=-1).contiguous().to(**to_float_args)
    batch["pcl"] = torch.stack(pcls_aug).contiguous().to(**to_float_args)


def get_rotation_torch(x_, y_, z_):
    x = (x_ / 180) * math.pi
    y = (y_ / 180) * math.pi
    z = (z_ / 180) * math.pi

    R_x = torch.tensor([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]], device=x_.device)
    R_y = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]], device=y_.device)
    R_z = torch.tensor([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]], device=z_.device)

    return torch.mm(R_z, torch.mm(R_y, R_x))


def get_init_scale_train(cfg, batch, device="cuda", dtype=torch.float32):
    tensor_kwargs = {"dtype": dtype, "device": device}
    input_cfg = cfg.INPUT
    n_obj = batch["obj_scale"].shape[0]
    init_pose_type = random.choice(input_cfg.INIT_SCALE_TYPE_TRAIN)  # randomly choose one type
    if init_pose_type == "gt_noise":
        batch["obj_scale_est"] = aug_scale_normal(
            batch["obj_scale"],
            std_scale=cfg.INPUT.NOISE_SCALE_STD_TRAIN,
            min_s=cfg.INPUT.INIT_SCALE_MIN,
        )
    elif init_pose_type == "random":  # random
        scale_rand = np.zeros((n_obj, 3), dtype="float32")
        for _i in range(n_obj):
            scale_rand[_i] = np.array([random.uniform(_min, _max) for _min, _max in zip(s_min, s_max)])
            s_min = input_cfg.RANDOM_SCALE_MIN
            s_max = input_cfg.RANDOM_SCALE_MAX
        batch["obj_scale_est"] = torch.tensor(scale_rand, **tensor_kwargs)
    elif init_pose_type == "last_frame":
        batch["obj_scale_est"] = batch["last_frame_poses"][:, :3, 4]
    elif init_pose_type == "canonical":
        s_canonical = torch.tensor(input_cfg.CANONICAL_SIZE, **tensor_kwargs).reshape(1, 3)
        batch["obj_scale_est"] = s_canonical.repeat(n_obj, 1)  # [n, 3]
    else:
        raise ValueError(f"Unknown init pose type for train: {init_pose_type}")


def get_init_pose_train(cfg, batch, device="cuda", dtype=torch.float32):
    tensor_kwargs = {"dtype": dtype, "device": device}
    input_cfg = cfg.INPUT
    n_obj = batch["obj_pose"].shape[0]
    init_pose_type = random.choice(input_cfg.INIT_POSE_TYPE_TRAIN)  # randomly choose one type
    if init_pose_type == "gt_noise":
        batch["obj_pose_est"] = aug_poses_normal(
            batch["obj_pose"],
            std_rot=input_cfg.NOISE_ROT_STD_TRAIN,  # randomly choose one
            std_trans=input_cfg.NOISE_TRANS_STD_TRAIN,  # [0.01, 0.01, 0.05]
            max_rot=input_cfg.NOISE_ROT_MAX_TRAIN,  # 45
            min_z=input_cfg.INIT_TRANS_MIN_Z,  # 0.1
        )
    elif init_pose_type == "random":  # random
        poses_rand = np.zeros((n_obj, 3, 4), dtype="float32")
        for _i in range(n_obj):
            poses_rand[_i, :3, :3] = random_rotation_matrix()[:3, :3]
            t_min = input_cfg.RANDOM_TRANS_MIN
            t_max = input_cfg.RANDOM_TRANS_MAX
            poses_rand[_i, :3, 3] = np.array([random.uniform(_min, _max) for _min, _max in zip(t_min, t_max)])
        batch["obj_pose_est"] = torch.tensor(poses_rand, **tensor_kwargs)
    elif init_pose_type == "last_frame":
        assert "last_frame_poses" in batch
        batch["obj_pose_est"] = batch["last_frame_poses"][:, :3, :4]
    elif init_pose_type == "canonical":
        r_canonical = rot_from_axangle_chain(input_cfg.CANONICAL_ROT)
        t_canonical = np.array(input_cfg.CANONICAL_TRANS)
        pose_canonical = torch.tensor(
            np.hstack([r_canonical, t_canonical.reshape(3, 1)]),
            **tensor_kwargs,
        )
        batch["obj_pose_est"] = pose_canonical.repeat(n_obj, 1, 1)  # [n,3,4]
    else:
        raise ValueError(f"Unknown init pose type for train: {init_pose_type}")


def _normalize_image(im, mean, std):
    # Bx3xHxW, 3x1x1
    return (im - mean) / std


def get_out_coor(cfg, coor_x, coor_y, coor_z):
    if (coor_x.shape[1] == 1) and (coor_y.shape[1] == 1) and (coor_z.shape[1] == 1):
        coor_ = torch.cat([coor_x, coor_y, coor_z], dim=1)
    else:
        coor_ = torch.stack(
            [torch.argmax(coor_x, dim=1), torch.argmax(coor_y, dim=1), torch.argmax(coor_z, dim=1)],
            dim=1,
        )
        # set the coordinats of background to (0, 0, 0)
        coor_[coor_ == cfg.MODEL.CATRE.XYZ_HEAD.XYZ_BIN] = 0
        # normalize the coordinates to [0, 1]
        coor_ = coor_ / float(cfg.MODEL.CATRE.XYZ_HEAD.XYZ_BIN - 1)

    return coor_


def get_out_mask(cfg, pred_mask):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    mask_loss_type = cfg.MODEL.CATRE.MASK_HEAD.MASK_LOSS_TYPE
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        out_mask = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-6)
    elif mask_loss_type == "BCE":
        assert c == 1, c
        out_mask = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        out_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
    return out_mask


def _zeros(_n, _c, _h, _w, dtype=torch.float32, device="cuda"):
    _tensor_kwargs = {"dtype": dtype, "device": device}
    return torch.zeros(_n, _c, _h, _w, **_tensor_kwargs).detach()


def _empty(_n, _c, _h, _w, dtype=torch.float32, device="cuda"):
    _tensor_kwargs = {"dtype": dtype, "device": device}
    return torch.empty(_n, _c, _h, _w, **_tensor_kwargs).detach()


def get_input_dim(cfg):
    backbone_cfg = cfg.MODEL.CATRE.BACKBONE
    if backbone_cfg.SHARED:
        return backbone_cfg.INIT_CFG.in_channels
    else:
        return backbone_cfg.INIT_CFG.in_channels // 2, backbone_cfg.INIT_CFG.in_channels // 2


def boxes_to_masks(boxes, imH, imW, device="cuda", dtype=torch.float32):
    n_obj = boxes.shape[0]
    masks = _zeros(n_obj, 1, imH, imW, device=device, dtype=dtype)  # the square region of bbox
    for _i in range(n_obj):
        x1, y1, x2, y2 = boxes[_i]
        x1 = int(min(imW - 1, max(0, x1)))
        y1 = int(min(imH - 1, max(0, y1)))
        x2 = int(min(imW - 1, max(0, x2)))
        y2 = int(min(imH - 1, max(0, y2)))
        masks[_i, 0, y1 : y2 + 1, x1 : x2 + 1] = 1.0
    return masks


def zoom_kps_batch(kps, K, K_zoom):
    invK = torch.linalg.inv(K)
    zoom_tfd_kps = invK @ K_zoom @ kps.clone().permute(0, 2, 1)
    return zoom_tfd_kps


def plot_3d(pcl1, pcl2, title):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection="3d"), figsize=plt.figaspect(1 / 3))
    axs[0].scatter(pcl1[:, 0], pcl1[:, 1], pcl1[:, 2], marker="o")
    axs[0].scatter(pcl2[:, 0], pcl2[:, 1], pcl2[:, 2], marker="x", color="red")

    axs[1].scatter(pcl1[:, 0], pcl1[:, 1], pcl1[:, 2], marker="o")
    axs[2].scatter(pcl2[:, 0], pcl2[:, 1], pcl2[:, 2], marker="x", color="red")

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(title)
    plt.show()


def vis_batch(cfg, batch, phase="train"):
    im_ids = batch["im_id"]
    n_obj = batch["obj_cls"].shape[0]
    Ks = batch["K"].detach().cpu().numpy()

    kpts_3d_list = batch["obj_kps"].detach().cpu().numpy()
    scale = batch["obj_scale"].detach().cpu().numpy()
    kpts_3d_list = kpts_3d_list * scale[:, None]
    kpts_3d_list = np.array([misc.get_3D_corners(kpts_3d_noise) for kpts_3d_noise in kpts_3d_list])
    pose_noise = batch["obj_pose_est"].detach().cpu().numpy()

    Rs_noise = pose_noise[:, :, :3]
    transes_noise = pose_noise[:, :, 3:]
    kpts_2d_noise = [
        misc.project_pts(kpt3d, K, R, t) for kpt3d, K, R, t in zip(kpts_3d_list, Ks, Rs_noise, transes_noise)
    ]

    poses = batch["obj_pose"].detach().cpu().numpy()
    scales = batch["obj_scale"].detach().cpu().numpy()
    pcl = batch["x"].permute(0, 2, 1).detach().cpu().numpy()
    tfd_kps = batch["tfd_kps"].permute(0, 2, 1).detach().cpu().numpy()
    nocses = batch["nocs"].detach().cpu().numpy()  # bs, 3, p
    Rs = poses[:, :, :3]
    transes = poses[:, :, 3:]
    kpts_2d_gt = [misc.project_pts(kpt3d, K, R, t) for kpt3d, K, R, t in zip(kpts_3d_list, Ks, Rs, transes)]
    # yapf: disable
    for i in range(n_obj):
        diag_len = np.linalg.norm(scales[i])
        R = Rs[i]
        t = transes[i]
        nocs_ = nocses[i] * diag_len
        nocs_ = (R @ nocs_ + t.reshape(3, 1)).T
        pcl_ = pcl[i] + transes_noise[i].reshape(1, 3)
        nocs_dist = np.linalg.norm(nocs_ - pcl_, axis=1).mean()
        plot_3d(pcl_, nocs_, f"pcl vs nocs, mean dist {nocs_dist}")
        vis_dict = {"img": (batch['img'][int(im_ids[i])].detach().cpu().numpy().transpose(1,2,0) * 255).astype('uint8')[:,:,::-1],
                    "depth": heatmap(batch['depth_obs'][int(im_ids[i])].detach().cpu().numpy().transpose(1,2,0), to_rgb=True)}
        img_vis_kpts2d_gt = misc.draw_projected_box3d(vis_dict["img"].copy(), kpts_2d_gt[i])
        vis_dict["img_vis_kpts2d_gt"] = img_vis_kpts2d_gt

        img_vis_kpts2d_noise = misc.draw_projected_box3d(vis_dict["img"].copy(), kpts_2d_noise[i])
        vis_dict["img_vis_kpts2d_noise"] = img_vis_kpts2d_noise

        show_titles = list(vis_dict.keys())
        show_ims = list(vis_dict.values())
        ncol = 2
        nrow = 2
        grid_show(show_ims, show_titles, row=nrow, col=ncol)
    # yapf: enable
