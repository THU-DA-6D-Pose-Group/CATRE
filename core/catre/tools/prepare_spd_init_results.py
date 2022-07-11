import numpy as np
import mmcv
import os.path as osp
import glob
import sys
from tqdm import tqdm
import setproctitle

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../../"))
sys.path.insert(0, PROJ_ROOT)
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle

setproctitle.setproctitle(osp.basename(__file__).split(".")[0])

data_root = osp.join(PROJ_ROOT, "datasets/NOCS")

# original spd results
spd_pose_dir = osp.join(PROJ_ROOT, "datasets/NOCS/deformnet_eval/eval_real")
spd_seg_dir = osp.join(PROJ_ROOT, "datasets/NOCS/deformnet_eval/mrcnn_results/real_test")

# our format
init_pose_dir = osp.join(data_root, "test_init_poses")
mmcv.mkdir_or_exist(init_pose_dir)
init_pose_path = osp.join(init_pose_dir, "init_pose_spd_nocs_real.json")


if __name__ == "__main__":
    results = {}

    CACHED = False
    if CACHED:
        results = mmcv.load(init_pose_path)
    else:
        spd_pose_paths = glob.glob(osp.join(spd_pose_dir, "results*.pkl"))

        num_total = 0
        for idx, spd_pose_path in enumerate(tqdm(spd_pose_paths)):
            preds = mmcv.load(spd_pose_path)
            bboxes = preds["pred_bboxes"]
            scores = preds["pred_scores"]
            poses = preds["pred_RTs"][:, :3]
            pred_scales = preds["pred_scales"]
            class_ids = preds["pred_class_ids"]
            mug_handles = preds["gt_handle_visibility"]

            scene_id, im_id = spd_pose_path.split("/")[-1].split(".")[0].split("_")[-2:]
            scene_im_id = f"scene_{scene_id}/{im_id}"

            seg_path = osp.join(spd_seg_dir, f"results_test_scene_{scene_id}_{im_id}.pkl")
            assert osp.exists(seg_path), seg_path

            masks = mmcv.load(seg_path)["masks"].astype("int")  # bool -> int
            assert masks.shape[2] == len(class_ids)
            results[scene_im_id] = []
            i = 0
            for class_id, pose, scale, score, bbox, mug_handle in zip(
                class_ids, poses, pred_scales, scores, bboxes, mug_handles
            ):
                # [sR -> R], normed_scale -> scale
                R = pose[:3, :3]
                nocs_scale = pow(np.linalg.det(R), 1/3)
                abs_scale = scale * nocs_scale
                pose[:3, :3] = R / nocs_scale
                # mask2rle
                mask = masks[:, :, i]
                mask_rle = binary_mask_to_rle(mask)
                y1, x1, y2, x2 = bbox.tolist()
                bbox = [x1, y1, x2, y2]
                cur_res = {
                    "obj_id": int(class_id),
                    "pose_est": pose.tolist(),
                    "scale_est": abs_scale.tolist(),
                    "bbox_est": bbox,
                    "score": float(score),
                    "mug_handle": int(mug_handle),
                    "segmentation": mask_rle,
                }
                results[scene_im_id].append(cur_res)
                i += 1

        print(init_pose_path)
        inout.save_json(init_pose_path, results, sort=False)

    VIS = False
    if VIS:
        from core.utils.data_utils import read_image_mmcv
        from lib.utils.mask_utils import cocosegm2mask
        from lib.vis_utils.image import grid_show, heatmap
        from core.catre.engine.test_utils import get_3d_bbox
        import ref

        for scene_im_id, r in results.items():
            img_path = f"datasets/NOCS/REAL/real_test/{scene_im_id}_color.png"
            img = read_image_mmcv(img_path, format="BGR")
            K = ref.nocs.real_intrinsics
            anno = r[0]
            imH, imW = img.shape[:2]
            mask = cocosegm2mask(anno["segmentation"], imH, imW)
            pose = np.array(anno["pose_est"]).reshape(3, 4)
            scale = np.array(anno["scale_est"])
            bbox = get_3d_bbox(scale).transpose()
            kpts_2d = misc.project_pts(bbox, K, pose[:, :3], pose[:, 3])
            img_vis_kpts2d = misc.draw_projected_box3d(img.copy(), kpts_2d)
            grid_show([img[:, :, ::-1], img_vis_kpts2d, mask], row=3, col=1)
