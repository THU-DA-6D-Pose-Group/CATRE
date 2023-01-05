import hashlib
import logging
import os
import os.path as osp
import copy
import sys
import time
from collections import OrderedDict
import mmcv
import numpy as np
from torch.utils.data import dataset
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import ref

from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, iprint, lazy_property

logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class CMRA_Dataset:
    """nocs split."""

    def __init__(self, data_cfg):
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects

        self.ann_files = data_cfg["ann_files"]  # idx files with image ids
        self.image_prefixes = data_cfg["image_prefixes"]

        self.dataset_root = data_cfg["dataset_root"]
        assert osp.exists(self.dataset_root), self.dataset_root
        self.model_path = data_cfg["model_path"]
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.models = mmcv.load(self.model_path)

        self.with_masks = data_cfg["with_masks"]  # True (load masks but may not use it)
        self.with_depth = data_cfg["with_depth"]  # True (load depth path here, but may not use it)
        self.with_coord = data_cfg["with_coord"]

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.img_type = data_cfg["img_type"]
        self.filter_invalid = data_cfg["filter_invalid"]

        self.mug_meta = self._get_mug_meta()

        self.cam = ref.cmra.intrinsics
        ##################################################

        self.catid2names = {cat_id: obj_name for cat_id, obj_name in ref.cmra.id2obj.items() if obj_name in self.objs}
        # NOTE: careful! Only the selected objects
        self.cat_ids = list(self.catid2names.keys())
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))

        self.bad_insts = [
            "1298634053ad50d36d07c55cf995503e",
            "2153bc743019671ae60635d9e388f801",
            "22217d5660444eeeca93934e5f39869",
            "290abe056b205c08240c46d333a693f",
            "39419e462f08dcbdc98cccf0d0f53d7",
            "4700873107186c6e2203f435e9e6785",
            "550aea46c75351a387cfe978d99ba05d",
            "60923e8a6c785a8755a834a7aafb0236",
            "6ed69b00b4632b6e07718ee10b83e10",
            "7077395b60bf4aeb3cb44973ec1ffcf8",
            "87b8cec4d55b5f2d75d556067d060edf",
            "97cd28c085e3754f22c69c86438afd28",
            "a9408583f2c4d6acad8a06dbee1d115",
            "b27815a2bde54ad3ab3dfa44f5fab01",
            "b42c73b391e14cb16f05a1f780f1cef",
            "c3e6564fe7c8157ecedd967f62b864ab",
            "c802792f388650428341191174307890",
            "d680d61f934eaa163b211460f022e3d9",
            "d9bb9c5a48c3afbfb84553e864d84802",
            "e3dc17dbde3087491a722bdf095986a4",
            "e57aa404a000df88d5d4532c6bb4bd2b",
            "eb86c8c2a20066d0fb1468f5fc754e02",
            "ee58b922bd93d01be4f112f1b3124b84",
            "fe669947912103aede650492e45fb14f",
            "ff74c4d2e710df3401a67448bf8fe08",
        ]
        ##########################################################

    def _get_mug_meta(self):
        data_ref = ref.cmra

        mug_meta_path = osp.join(data_ref.model_dir, "mug_meta.pkl")
        mug_meta = mmcv.load(mug_meta_path)
        return mug_meta

    def _load_from_idx_file(self, idx_file, image_root):
        """return dataset_dict.

        Args:
            idx_file ([str]): path to images
        """
        split_scene_im_ids = []
        gt_dicts = {}
        with open(idx_file, "r") as f:
            for line in f:
                split_scene_im_id = line.strip("\r\n")

                label_path = osp.join(image_root, f"{split_scene_im_id}_label.pkl")
                try:
                    gt_dict = mmcv.load(label_path)
                except:
                    continue
                gt_dicts[split_scene_im_id] = gt_dict
                split_scene_im_ids.append(split_scene_im_id)

        if "train_real" in self.name:
            mug_handle_dict = mmcv.load(osp.join(image_root, "real_train/mug_handle.pkl"))
        else:
            mug_handle_dict = None

        split_scene_im_ids = sorted(split_scene_im_ids)
        dataset_dicts = []

        num_instances_without_valid_segmentation = 0
        num_instances_without_valid_box = 0
        num_instance_without_mask_file = 0
        num_instance_without_depth_file = 0
        num_instance_without_coord_file = 0
        num_instance_without_img_file = 0

        for split_scene_im_id in tqdm(split_scene_im_ids):
            split, scene_id, im_id = split_scene_im_id.split("/")
            scene_im_id = f"{scene_id}/{im_id}"  # e.g. , scene_1/0000
            gt_dict = gt_dicts[split_scene_im_id]

            rgb_path = osp.join(image_root, f"{split_scene_im_id}_color.png")
            coord_path = osp.join(image_root, f"{split_scene_im_id}_coord.png")
            depth_path = osp.join(image_root, f"{split_scene_im_id}_depth.png")
            mask_path = osp.join(image_root, f"{split_scene_im_id}_mask.png")

            if not os.access(rgb_path, os.R_OK):
                logger.info(f"img file {rgb_path} can not be accessed!")
                num_instance_without_img_file += 1
                continue

            record = {
                "dataset_name": self.name,
                "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                "height": self.height,
                "width": self.width,
                "scene_im_id": scene_im_id,  # for evaluation
                "cam": self.cam,  # self.cam,
                "img_type": self.img_type,
            }

            if self.with_depth:
                if not os.access(depth_path, os.R_OK):
                    logger.info(f"depth file {depth_path} can not be accessed!")
                    num_instance_without_depth_file += 1
                    continue
                record["depth_file"] = depth_path

            if self.with_coord:
                if not os.access(coord_path, os.R_OK):
                    logger.info(f"coord file {coord_path} can not be accessed!")
                    num_instance_without_coord_file += 1
                    continue
                record["coord_file"] = coord_path

            if self.with_masks:
                if not os.access(mask_path, os.R_OK):
                    logger.info(f"mask file {mask_path} can not be accessed!")
                    num_instance_without_mask_file += 1
                    continue
                mask_full = mmcv.imread(mask_path, "unchanged")
                if mask_full is None:
                    logger.info(f"mask file {mask_path} can not be accessed!")
                    num_instance_without_mask_file += 1
                    continue

            # instance-level annotations
            insts = []
            # NOTE: idx and inst_id are not exactly matched!
            for idx, inst_id in enumerate(gt_dict["instance_ids"]):
                class_id = gt_dict["class_ids"][idx]

                if class_id not in self.cat_ids:
                    continue

                obj_name = self.catid2names[class_id]  # category label
                inst_name = gt_dict["model_list"][idx]  # instance label

                if inst_name in self.bad_insts:
                    continue

                if obj_name == "mug" and mug_handle_dict is not None:
                    mug_handle = mug_handle_dict[f"{scene_id}_res"][int(im_id)]
                else:
                    mug_handle = 1

                # 0-based label now
                cur_label = self.cat2label[class_id]
                ################ pose ###########################
                R = gt_dict["rotations"][idx]
                trans = gt_dict["translations"][idx]

                ################ scale ###########################
                nocs_scale = gt_dict["scales"][idx]

                assert inst_name in self.models
                model = self.models[inst_name]
                abs_scale = self.get_abs_scale(model, nocs_scale, obj_name)  # m

                ################ pose ###########################
                pose = np.hstack([R, trans.reshape(3, 1)])
                quat = mat2quat(pose[:3, :3])

                ############# bbox ############################
                y1, x1, y2, x2 = gt_dict["bboxes"][idx]  # yxyx
                bbox = [x1, y1, x2, y2]

                if self.filter_invalid:
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    if bh <= 1 or bw <= 1:
                        num_instances_without_valid_box += 1
                        continue

                ############## mask #######################
                if self.with_masks:
                    if len(mask_full.shape) == 3:
                        mask_full = mask_full[:, :, 2]
                    assert len(mask_full.shape) == 2
                    mask = np.zeros_like(mask_full)
                    mask[mask_full == inst_id] = 1  # fg mask, 0/1
                    area = mask.sum()
                    if area < 30 and self.filter_invalid:
                        num_instances_without_valid_segmentation += 1
                        continue
                    mask_rle = binary_mask_to_rle(mask)

                proj = (self.cam @ trans.T).T  # NOTE: use self.cam here
                proj = proj[:2] / proj[2]

                inst = {
                    "category_id": cur_label,  # 0-based label
                    "inst_name": inst_name,
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "pose": pose,
                    "quat": quat,
                    "trans": trans,
                    "scale": abs_scale,  # m
                    "centroid_2d": proj,  # absolute (cx, cy)
                    "segmentation": mask_rle,
                    "mug_handle": mug_handle,
                    "nocs_scale": nocs_scale,
                }
                for key in ["bbox3d_and_center"]:
                    inst[key] = self.get_bbox3d_and_center(model, nocs_scale)
                insts.append(inst)

            if len(insts) == 0:
                continue
            record["annotations"] = insts
            dataset_dicts.append(record)

        if num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    num_instances_without_valid_segmentation
                )
            )
        if num_instances_without_valid_box > 0:
            logger.warning(
                "Filtered out {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(num_instances_without_valid_box)
            )
        return dataset_dicts

    def __call__(self):  # NOCS
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name,
                    self.dataset_root,
                    self.with_masks,
                    self.with_depth,
                    self.with_coord,
                    __name__,
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(
            self.cache_dir,
            "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name),
        )

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        logger.info("loading dataset dicts: {}".format(self.name))
        t_start = time.perf_counter()
        dataset_dicts = []
        self._unique_im_id = 0
        for ann_file, image_root in zip(self.ann_files, self.image_prefixes):
            dataset_dicts.extend(self._load_from_idx_file(ann_file, image_root))

        ##########################################################################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    def get_abs_scale(self, model, nocs_scale, obj_name):
        # model pc x 3, normalized scale
        # lx = max(model[:, 0]) - min(model[:, 0])
        if obj_name == "mug":
            lx = 2 * max(max(model[:, 0]), -min(model[:, 0]))
        else:
            lx = max(model[:, 0]) - min(model[:, 0])

        ly = max(model[:, 1]) - min(model[:, 1])
        lz = max(model[:, 2]) - min(model[:, 2])

        # absolutely scale(m)
        lx_t = lx * nocs_scale
        ly_t = ly * nocs_scale
        lz_t = lz * nocs_scale

        abs_scale = np.array([lx_t, ly_t, lz_t], dtype=np.float32)

        return abs_scale

    def get_bbox3d_and_center(self, pts, nocs_scale):
        minx, maxx = min(pts[:, 0]), max(pts[:, 0])
        miny, maxy = min(pts[:, 1]), max(pts[:, 1])
        minz, maxz = min(pts[:, 2]), max(pts[:, 2])
        avgx = np.average(pts[:, 0])
        avgy = np.average(pts[:, 1])
        avgz = np.average(pts[:, 2])
        # NOTE: we use a different order from roi10d
        """
              1 -------- 0
             /|         /|
            2 -------- 3 .
            | |        | |
            . 5 -------- 4
            |/         |/
            6 -------- 7
        """
        bb = (
            np.array(
                [
                    [maxx, maxy, maxz],
                    [minx, maxy, maxz],
                    [minx, miny, maxz],
                    [maxx, miny, maxz],
                    [maxx, maxy, minz],
                    [minx, maxy, minz],
                    [minx, miny, minz],
                    [maxx, miny, minz],
                    [avgx, avgy, avgz],
                ],
                dtype=np.float32,
            )
            * nocs_scale
        )
        return bb


default_cfg = dict(
    scale_to_meter=0.001,
    with_masks=True,  # (load masks but may not use it)
    with_depth=True,  # (load depth path here, but may not use it)
    with_coord=True,  # load nocs here
    height=480,
    width=640,
    cache_dir=osp.join(PROJ_ROOT, ".cache"),
    use_cache=True,
    num_to_load=-1,
    filter_scene=True,
    filter_invalid=True,
    ref_key="cmra",
)
SPLITS_NOCS = {}

update_cfgs = dict(
    nocs_train_cmra=dict(
        objs=ref.cmra.objects,
        dataset_root=osp.join(DATASETS_ROOT, "NOCS/CAMERA/train"),
        model_path=osp.join(DATASETS_ROOT, "NOCS/obj_models/camera_train.pkl"),
        ann_files=[
            osp.join(
                DATASETS_ROOT,
                "NOCS/CAMERA/image_set/train_list.txt",
            )
        ],
        img_type="camera",
        image_prefixes=[osp.join(DATASETS_ROOT, "NOCS/CAMERA/")],
    ),
    nocs_train_cmra_part2=dict(
        objs=["camera", "can", "laptop", "mug"],
        dataset_root=osp.join(DATASETS_ROOT, "NOCS/CAMERA/train"),
        model_path=osp.join(DATASETS_ROOT, "NOCS/obj_models/camera_train.pkl"),
        ann_files=[
            osp.join(
                DATASETS_ROOT,
                "NOCS/CAMERA/image_set/train_list.txt",
            )
        ],
        img_type="camera",
        image_prefixes=[osp.join(DATASETS_ROOT, "NOCS/CAMERA/")],
    ),
    nocs_val_cmra=dict(
        objs=ref.cmra.objects,
        dataset_root=osp.join(DATASETS_ROOT, "NOCS/CAMERA/val"),
        model_path=osp.join(
            DATASETS_ROOT, "NOCS/obj_models/camera_val.pkl"
        ),  # NOTE: this is not allowed to use during inference!
        ann_files=[
            osp.join(
                DATASETS_ROOT,
                "NOCS/CAMERA/image_set/val_list.txt",
            )
        ],
        img_type="camera",
        image_prefixes=[osp.join(DATASETS_ROOT, "NOCS/CAMERA/")],
    ),
)

# single object splits ######################################################
for obj in ref.cmra.objects:
    for split in ["train_cmra", "val_cmra"]:
        mode = split.replace("_cmra", "")  # train/test/val
        name = f"nocs_{obj}_{split}"
        if mode == "train":
            filter_invalid = True
        else:
            filter_invalid = False

        if name not in SPLITS_NOCS:
            update_cfgs[name] = dict(
                objs=[obj],
                dataset_root=osp.join(DATASETS_ROOT, f"NOCS/CAMERA/{mode}"),
                model_path=osp.join(DATASETS_ROOT, f"NOCS/obj_models/camera_{mode}.pkl"),
                ann_files=[
                    osp.join(
                        DATASETS_ROOT,
                        f"NOCS/CAMERA/image_set/{obj}_CAMERA_{mode}.txt",
                    )
                ],
                image_prefixes=[osp.join(DATASETS_ROOT, f"NOCS/CAMERA/")],
                img_type="camera",
            )

for name, update_cfg in update_cfgs.items():
    used_cfg = copy.deepcopy(default_cfg)
    used_cfg["name"] = name
    used_cfg.update(update_cfg)
    SPLITS_NOCS[name] = used_cfg


def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_NOCS:
        used_cfg = SPLITS_NOCS[name]
    else:
        assert (
            data_cfg is not None
        ), f"dataset name {name} is not registered. available datasets: {list(SPLITS_NOCS.keys())}"
        used_cfg = data_cfg
    DatasetCatalog.register(name, CMRA_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="nocs",
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["rete", "iou"],
        evaluator_type="nocs",
    )


def get_available_datasets():
    return list(SPLITS_NOCS.keys())


def test_vis():
    # NOTE: coord are not checked!
    dataset_name = sys.argv[1]
    meta = MetadataCatalog.get(dataset_name)
    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dataset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))
    objs = meta.objs
    for d in dicts:
        img = read_image_mmcv(d["file_name"], format="BGR")
        depth = load_depth(d["depth_file"]) / 1000.0

        imH, imW = img.shape[:2]
        annos = d["annotations"]
        masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
        bboxes = [anno["bbox"] for anno in annos]
        bbox_modes = [anno["bbox_mode"] for anno in annos]
        bboxes_xyxy = np.array(
            [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
        )
        # kpts_3d_list = [anno["bbox3d_and_center"] for anno in annos]
        quats = [anno["quat"] for anno in annos]
        transes = [anno["trans"] for anno in annos]
        kpts_3d_list = [get_bbox_from_scale(anno["scale"]) for anno in annos]
        Rs = [quat2mat(quat) for quat in quats]
        # 0-based label
        cat_ids = [anno["category_id"] for anno in annos]
        K = d["cam"]
        kpts_2d = [misc.project_pts(kpt3d, K, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)]
        labels = [objs[cat_id] for cat_id in cat_ids]
        for _i in range(len(annos)):
            img_vis = vis_image_mask_bbox_cv2(
                img,
                masks[_i : _i + 1],
                bboxes=bboxes_xyxy[_i : _i + 1],
                labels=labels[_i : _i + 1],
            )
            img_vis_kpts2d = misc.draw_projected_box3d(img_vis.copy(), kpts_2d[_i])

            grid_show(
                [
                    img[:, :, [2, 1, 0]],
                    img_vis[:, :, [2, 1, 0]],
                    img_vis_kpts2d[:, :, [2, 1, 0]],
                    depth,
                ],
                ["img", "vis_img", "img_vis_kpts2d", "depth"],
                row=2,
                col=2,
            )


if __name__ == "__main__":
    """Test the  dataset loader.

    Usage:
        python -m this_module dataset_name
        "dataset_name" can be any pre-registered ones
    """
    from lib.vis_utils.image import grid_show
    from lib.utils.setup_logger import setup_my_logger

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from core.utils.data_utils import read_image_mmcv
    from core.utils.cat_data_utils import get_bbox_from_scale, load_depth

    print("sys.argv:", sys.argv)
    logger = setup_my_logger(name="core")
    register_with_name_cfg(sys.argv[1])
    print("dataset catalog: ", DatasetCatalog.list())
    test_vis()
