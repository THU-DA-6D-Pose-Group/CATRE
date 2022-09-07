from re import match
import numpy as np
import mmcv
import os.path as osp
import sys
from tqdm import tqdm
import setproctitle

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../"))
sys.path.insert(0, PROJ_ROOT)

import ref
from lib.utils.mask_utils import binary_mask_to_rle
from lib.pysixd.misc import iou

setproctitle.setproctitle(osp.basename(__file__).split(".")[0])

data_root = osp.join(PROJ_ROOT, "datasets/NOCS")

# original dualposenet results
dpn_pose_dir = osp.join(PROJ_ROOT, "datasets/NOCS/dualposenet_results")  # provided by the authors
spd_pose_dir = osp.join(PROJ_ROOT, "datasets/NOCS/REAL/real_test")


def _get_mug_meta():
    mug_meta_path = osp.join(ref.nocs.model_dir, "mug_meta.pkl")
    mug_meta = mmcv.load(mug_meta_path)
    return mug_meta


def read_meta(path):
    with open(path) as f:
        names = {}
        for line in f.readlines():
            _, cls, name = line.strip("\n").split(" ")
            names[cls] = name
    return names


def yxyx_to_xywh(bbox):
    y1, x1, y2, x2 = bbox
    w1 = x2 - x1
    h1 = y2 - y1
    return [x1, y1, w1, h1]


def find_best_match_for_pred(gt_class_ids, pred_class_ids, gt_bboxes, pred_bboxes):
    match_ids = []
    pred_idx = np.where(pred_class_ids == 6)[0]
    gt_idx = np.where(gt_class_ids == 6)[0]

    if len(pred_idx) == 1 and len(gt_idx) == 1:
        match_id = [pred_idx[0], gt_idx[0]]
        match_ids.append(match_id)
    else:
        for pidx in pred_idx:
            pred_bbox = pred_bboxes[pidx]
            pbox = yxyx_to_xywh(pred_bbox)
            max_iou = -1
            match_id = [pidx, None]
            for gidx in gt_idx:
                gt_bbox = gt_bboxes[gidx]
                gbox = yxyx_to_xywh(gt_bbox)
                _iou = iou(pbox, gbox)
                if _iou > max_iou:
                    max_iou = _iou
                    match_id[1] = gidx
            if not max_iou > 0:
                match_id[1] = None
            match_ids.append(match_id)

    return match_ids


if __name__ == "__main__":
    dpn_pose_path = osp.join(dpn_pose_dir, "REAL275_results.pkl")
    dpn_new_pose_path = osp.join(dpn_pose_dir, "REAL275_results_mvmug.pkl")
    dpn_pose_res = mmcv.load(dpn_pose_path)

    # mug_meta = _get_mug_meta()
    new_res = []

    for idx, preds in enumerate(tqdm(dpn_pose_res)):

        scene_id, im_id = preds["image_path"].split("/")[-2:]

        # two special cases
        if scene_id == "scene_1" and im_id == "0205":
            for key in ["gt_class_ids", "gt_bboxes", "gt_RTs", "gt_scales", "gt_handle_visibility"]:
                preds[key] = preds[key][1:]

        if scene_id == "scene_1" and im_id == "0197":
            for key in ["gt_class_ids", "gt_bboxes", "gt_RTs", "gt_scales", "gt_handle_visibility"]:
                preds[key] = preds[key][[0, 1, 2, 4]]

        new_preds = preds.copy()
        scene_im_id = f"{scene_id}/{im_id}"

        spd_label_path = osp.join(spd_pose_dir, scene_id, f"{im_id}_label.pkl")
        spd_label = mmcv.load(spd_label_path)

        meta_info_path = osp.join(spd_pose_dir, scene_id, f"{im_id}_meta.txt")
        meta = read_meta(meta_info_path)

        spd_class_ids = spd_label["class_ids"]
        spd_poses = spd_label["poses"].astype(np.float32)
        spd_nocs_scales = spd_label["scales"].astype(np.float32)
        spd_scales = spd_label["size"].astype(np.float32)

        # find matches for preds
        gt_class_ids = preds["gt_class_ids"]
        pred_class_ids = preds["pred_class_ids"]

        assert (spd_class_ids == gt_class_ids).all()

        gt_bboxes = preds["gt_bboxes"]
        pred_bboxes = preds["pred_bboxes"]

        gt_poses = preds["gt_RTs"]
        gt_scales = preds["gt_scales"]

        pred_poses = preds["pred_RTs"]
        pred_scales = preds["pred_scales"]

        # NOTE: choose best match by 2d bbox iou
        matchs = find_best_match_for_pred(gt_class_ids, pred_class_ids, gt_bboxes, pred_bboxes)

        for m in matchs:
            pid, gid = m

            if gid is None:
                continue

            gt_pose = gt_poses[gid]
            pred_pose = pred_poses[pid]
            gt_bbox = gt_bboxes[gid]
            gt_scale = gt_scales[gid]
            pred_scale = pred_scales[pid]

            spd_pose = spd_poses[gid]
            spd_nocs_scale = spd_nocs_scales[gid]  # nocs_scale
            spd_scale = spd_scales[gid]

            s0 = np.mean(gt_pose[:3, :3] / spd_pose[:3, :3])
            nocs_scale = spd_nocs_scale * s0

            new_pred_pose = np.identity(4, dtype=np.float32)
            new_pred_r = pred_pose[:3, :3] / s0
            delta_t = spd_pose[:3, 3] - gt_pose[:3, 3]
            new_pred_t = pred_pose[:3, 3] + delta_t

            new_pred_pose[:3, :3] = new_pred_r
            new_pred_pose[:3, 3] = new_pred_t

            pred_abscale = nocs_scale * pred_scale
            gt_abscale = nocs_scale * gt_scale
            spd_abscale = spd_nocs_scale * spd_scale

            delta_s = spd_abscale - gt_abscale
            new_pred_absscale = pred_abscale + delta_s

            new_pred_scale = new_pred_absscale / np.linalg.norm(new_pred_absscale)

            new_preds["gt_RTs"][gid] = spd_pose
            new_preds["pred_RTs"][pid] = new_pred_pose
            new_preds["gt_scales"][gid] = spd_scale
            new_preds["pred_scales"][pid] = new_pred_scale

        new_res.append(new_preds)

    mmcv.dump(new_res, dpn_new_pose_path)
