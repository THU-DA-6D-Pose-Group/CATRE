# -*- coding: utf-8 -*-
import copy
import logging
import os
import os.path as osp
import pickle

import cv2
import mmcv
import numpy as np
import ref
import torch

from core.base_data_loader import Base_DatasetFromList
from lib.structures import Center2Ds, Keypoints2Ds, Keypoints3Ds, MyBitMasks, Translations, Poses, MyList
from core.utils.data_utils import read_image_mmcv
from core.utils.dataset_utils import (
    filter_empty_dets,
    filter_invalid_in_dataset_dicts,
    flat_dataset_dicts,
    load_catre_init_into_dataset,
    my_build_batch_data_loader,
    trivial_batch_collator,
)
from core.utils.pose_aug import aug_poses_normal_np, aug_scale_normal_np
from core.utils.cat_data_utils import (
    load_depth,
    crop_ball_from_depth_image,
    crop_mask_depth_image,
    occlude_obj_by_bboxes,
)
from core.utils.pose_utils import rot_from_axangle_chain
from core.utils.my_distributed_sampler import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from core.utils.ssd_color_transform import ColorAugSSDTransform
from core.utils.depth_aug import add_noise_depth
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances, Keypoints, PolygonMasks
from detectron2.utils.logger import log_first_n
from lib.pysixd import inout, misc
from lib.utils.mask_utils import cocosegm2mask, get_edge
from lib.vis_utils.image import grid_show, heatmap
from einops import rearrange
from .dataset_factory import register_datasets

logger = logging.getLogger(__name__)


def build_catre_augmentation(cfg, is_train):
    """Create a list of :class:`Augmentation` from config. when training 6d
    pose, cannot flip.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    augmentation = []
    augmentation.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        # augmentation.append(T.RandomFlip())
        logger.info("Augmentations used in training: " + str(augmentation))
    return augmentation


def transform_instance_annotations(
    annotation,
    transforms,
    image_size,
    *,
    keypoint_hflip_indices=None,
    bbox_key="bbox",
):
    """
    NOTE: Adapted from detection_utils.
    Apply transforms to box, segmentation, keypoints, etc. of annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields `bbox_key`, "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    im_H, im_W = image_size
    bbox = BoxMode.convert(annotation[bbox_key], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation[bbox_key] = np.array(transforms.apply_box([bbox])[0])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # NOTE: here we transform segms to binary masks (interp is nearest by default)
        if isinstance(annotation["segmentation"], np.ndarray):
            mask_ori = annotation["segmentation"]
        else:
            mask_ori = cocosegm2mask(annotation["segmentation"], h=im_H, w=im_W)
        mask = transforms.apply_segmentation(mask_ori)
        annotation["segmentation"] = mask  # NOTE: visib_mask
        if "trunc_mask" not in annotation:
            annotation["trunc_mask"] = mask.copy()

    # TODO: maybe also load obj_masks (full masks)
    if "trunc_mask" in annotation:
        annotation["trunc_mask"] = transforms.apply_segmentation(annotation["trunc_mask"])

    if "keypoints" in annotation:
        keypoints = utils.transform_keypoint_annotations(
            annotation["keypoints"],
            transforms,
            image_size,
            keypoint_hflip_indices,
        )
        annotation["keypoints"] = keypoints

    if "centroid_2d" in annotation:
        annotation["centroid_2d"] = transforms.apply_coords(np.array(annotation["centroid_2d"]).reshape(1, 2)).flatten()

    return annotation


def annotations_to_instances(cfg, annos, image_size, mask_format="bitmask", K=None):
    """# NOTE: modified from detection_utils Create an :class:`Instances`
    object used by the models, from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields:
                "obj_classes",
                "obj_poses",
                "obj_boxes",
                "obj_boxes_det",  (used for keep original detected bbox xyxy)
                "obj_masks",
                "obj_3d_points"
            if they can be obtained from `annos`.
    """
    insts = Instances(image_size)  # (h,w)

    if all("bbox" in obj for obj in annos):  # not necessary when training
        boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        insts = Instances(image_size)  # (h,w)
        boxes = insts.obj_boxes = Boxes(np.array(boxes))
        boxes.clip(image_size)

    if all("bbox_det" in obj for obj in annos):
        boxes_det = [BoxMode.convert(obj["bbox_det"], obj["bbox_det_mode"], BoxMode.XYXY_ABS) for obj in annos]
        insts.obj_boxes_det = Boxes(np.array(boxes_det))
        insts.obj_boxes_det.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    # some properties do not need a new structure
    classes = torch.tensor(classes, dtype=torch.int64)
    insts.obj_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            # NOTE: should be kept on cpu, otherwise CUDA out of memory?
            masks = MyBitMasks(torch.stack([torch.tensor(x.copy()) for x in segms]), cpu_only=True)

        insts.obj_visib_masks = masks

    if len(annos) and "trunc_mask" in annos[0]:
        segms = [obj["trunc_mask"] for obj in annos]
        # NOTE: should be kept on cpu, otherwise CUDA out of memory?
        masks = MyBitMasks(torch.stack([torch.tensor(x.copy()) for x in segms]), cpu_only=True)
        insts.obj_trunc_masks = masks

    # for symmertries
    if len(annos) and "mug_handle" in annos[0]:
        mug_handles = [obj["mug_handle"] for obj in annos]
        insts.mug_handles = torch.tensor(mug_handles, dtype=bool)

    # for kps points and mug nocs map
    if len(annos) and "inst_name" in annos[0]:  # for test
        inst_names = [obj["inst_name"] for obj in annos]
        insts.inst_names = inst_names

    # NOTE: pose/scale related annotations
    # for train: this is gt pose/scale
    # for test: this is init pose/scale
    if len(annos) and "pose" in annos[0]:
        poses = [obj["pose"] for obj in annos]
        insts.obj_poses = Poses(np.array(poses))

    if len(annos) and "scale" in annos[0]:
        scales = [obj["scale"] for obj in annos]
        insts.obj_scales = torch.tensor(np.array(scales), dtype=torch.float32)

    if len(annos) and "score" in annos[0]:  # for test
        scores = [obj["score"] for obj in annos]
        insts.obj_scores = torch.tensor(scores, dtype=torch.float32)

    return insts


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.obj_boxes.nonempty(threshold=box_threshold))
    if instances.has("obj_visib_masks") and by_mask:
        r.append(instances.obj_visib_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x  # bool tensor

    return instances[m]


class CATRE_DatasetFromList(Base_DatasetFromList):
    """NOTE: we can also use the default DatasetFromList and
    implement a similar custom DataMapper,
    but it is harder to implement some features relying on other dataset dicts
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/common.py
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, cfg, split, lst: list, copy: bool = True, serialize: bool = True, flatten=False):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool): whether to hold memory using serialized objects, when
                enabled, data loader workers can use shared RAM from master
                process instead of making a copy.
        """
        self.augmentation = build_catre_augmentation(cfg, is_train=(split == "train"))
        if cfg.INPUT.COLOR_AUG_PROB > 0 and cfg.INPUT.COLOR_AUG_TYPE.lower() == "ssd":
            self.augmentation.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            logging.getLogger(__name__).info("Color augmentation used in training: " + str(self.augmentation[-1]))
        # fmt: off
        self.img_format = cfg.INPUT.FORMAT  # default BGR

        self.with_depth = cfg.INPUT.WITH_DEPTH
        self.aug_depth = cfg.INPUT.AUG_DEPTH
        self.drop_depth_ratio = cfg.INPUT.DROP_DEPTH_RATIO
        self.drop_depth_prob = cfg.INPUT.DROP_DEPTH_PROB
        self.add_noise_depth_level = cfg.INPUT.ADD_NOISE_DEPTH_LEVEL
        self.add_noise_depth_prob = cfg.INPUT.ADD_NOISE_DEPTH_PROB
        self.with_bg_depth = cfg.INPUT.WITH_BG_DEPTH

        # NOTE: color augmentation config
        self.color_aug_prob = cfg.INPUT.COLOR_AUG_PROB
        self.color_aug_type = cfg.INPUT.COLOR_AUG_TYPE
        self.color_aug_code = cfg.INPUT.COLOR_AUG_CODE
        # fmt: on
        self.cfg = cfg
        self.split = split  # train | val | test
        if split == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        # ------------------------
        # common model infos
        self.key_points = {}
        self.model_points = {}
        # -------------------------------------------------
        self.with_last_poses = "last_frame" in cfg.INPUT.INIT_POSE_TYPE_TRAIN
        if self.with_last_poses:
            self.last_frame_pose_dict = mmcv.load(cfg.INPUT.INIT_POSE_TRAIN_PATH)

        # ----------------------------------------------------
        self.mean_model_dict = self._load_mean_model_points(cfg.INPUT.MEAN_MODEL_PATH)

        # ----------------------------------------------------
        self.flatten = flatten
        self._lst = flat_dataset_dicts(lst) if flatten else lst
        # ----------------------------------------------------
        self._copy = copy
        self._serialize = serialize

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger.info("Serializing {} elements to byte tensors and concatenating them all ...".format(len(self._lst)))
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        if self._serialize:
            return len(self._addr)
        else:
            return len(self._lst)

    # NOTE: here we use gt model in the test set, so this is just an ablation setting!
    def _get_fps_points(self, dataset_name, inst_name, with_center=False):
        """return fps points."""
        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        objs = dset_meta.objs
        cfg = self.cfg
        num_keypoints = cfg.INPUT.NUM_KPS
        loaded_keypoints = data_ref.get_fps_points()
        if with_center:
            cur_keypoints = loaded_keypoints[inst_name][f"fps{num_keypoints}_and_center"]
        else:
            cur_keypoints = loaded_keypoints[inst_name][f"fps{num_keypoints}_and_center"][:-1]

        return cur_keypoints

    def _get_mug_meta(self, dataset_name):
        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]

        mug_meta_path = osp.join(data_ref.model_dir, "mug_meta.pkl")
        self.mug_meta = mmcv.load(mug_meta_path)

    def _get_mean_scale(self, dataset_name):
        dset_meta = MetadataCatalog.get(dataset_name)
        objs = dset_meta.objs
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]

        mean_scale_dict = data_ref.mean_scale

        self.mean_scales = []
        for obj in objs:
            self.mean_scales.append(mean_scale_dict[obj])

    def _load_mean_model_points(self, path, shuffle=True):
        mean_model_dicts = mmcv.load(path)

        if shuffle:
            for inst_name, pts in mean_model_dicts.items():
                num = pts.shape[0]
                keep_idx = np.arange(num)
                np.random.shuffle(keep_idx)
                mean_model_dicts[inst_name] = pts[keep_idx]

        return mean_model_dicts

    def _get_sym_infos(self, dataset_name, obj_id, mug_handle):

        dset_meta = MetadataCatalog.get(dataset_name)
        ref_key = dset_meta.ref_key
        data_ref = ref.__dict__[ref_key]
        obj_name = dset_meta.objs[obj_id]

        sym_axis_info = data_ref.get_sym_info(obj_name, mug_handle)
        if sym_axis_info is not None:
            sym_transforms = misc.get_axis_symmetry_transformations(
                sym_axis_info, max_sym_disc_step=self.cfg.INPUT.MAX_SYM_DISC_STEP
            )
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
        else:
            return None, obj_name

        return sym_info, obj_name

    def read_data_train(self, dataset_dict):
        """load image and annos random shift & scale bbox; crop, rescale."""
        assert self.split == "train", self.split
        cfg = self.cfg
        input_cfg = cfg.INPUT
        net_cfg = cfg.MODEL.CATRE

        backbone_cfg = net_cfg.BACKBONE
        pcl_net_cfg = net_cfg.PCLNET

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        dataset_name = dataset_dict["dataset_name"]
        scene_im_id = dataset_dict["scene_im_id"]

        image = read_image_mmcv(dataset_dict["file_name"], format=self.img_format)
        # should be consistent with the size in dataset_dict
        utils.check_image_size(dataset_dict, image)
        im_H_ori, im_W_ori = image.shape[:2]

        if "cam" in dataset_dict:
            K = dataset_dict["cam"].astype("float32")
            dataset_dict["cam"] = torch.as_tensor(K)
        else:
            raise RuntimeError("cam intrinsic is missing")

        # load mug meta information =============================================
        self._get_mug_meta(dataset_name)
        self._get_mean_scale(dataset_name)

        # load nocs(coordinate) =================================================
        coord_path = dataset_dict["coord_file"]
        coord = mmcv.imread(coord_path)[:, :, :3]  # 0 - 255
        coord = coord[:, :, (2, 1, 0)]
        coord = np.array(coord, dtype=np.float32) / 255
        coord[:, :, 2] = 1 - coord[:, :, 2]
        coord = coord - 0.5  # [0, 1] -> [-0.5, 0.5]
        coord = torch.from_numpy(coord)

        # load depth =================================================
        in_depth_channels = 0
        if self.with_depth:
            assert "depth_file" in dataset_dict, "depth file is not in dataset_dict"
            depth_path = dataset_dict["depth_file"]
            log_first_n(logging.WARN, "with depth", n=1)
            depth = load_depth(depth_path)  # uint16
            depth = depth.astype("float32") / 1000.0  # to m

            if cfg.INPUT.BP_DEPTH:
                in_depth_channels = 3
                depth = misc.backproject(depth, K)
            else:
                in_depth_channels = 1
                depth = depth[:, :, None]  # hwc
            # grid_show([heatmap(depth_ori, to_rgb=True), heatmap(depth, to_rgb=True)], ["depth_bp", "depth"], row=1, col=2)

        # currently only replace bg for train ###############################
        # NOTE: nocs camera data already has bg
        img_type = dataset_dict.get("img_type", "real")  # real/camera
        do_replace_bg = False
        if img_type == "syn":
            log_first_n(logging.WARNING, "replace bg", n=10)
            do_replace_bg = True
        else:  # real image or nocs camera
            if cfg.INPUT.WITH_IMG and np.random.rand() < cfg.INPUT.CHANGE_BG_PROB:
                log_first_n(logging.WARNING, f"replace bg for {img_type}", n=10)
                do_replace_bg = True
        if do_replace_bg:
            assert "annotations" in dataset_dict and "segmentation" in dataset_dict["annotations"][0]
            _trunc_fg = cfg.INPUT.get("TRUNCATE_FG", False)
            for anno_i, anno in enumerate(dataset_dict["annotations"]):
                anno["segmentation"] = visib_mask = cocosegm2mask(anno["segmentation"], im_H_ori, im_W_ori)
                if _trunc_fg:
                    anno["trunc_mask"] = self.trunc_mask(visib_mask).astype("uint8")

            if _trunc_fg:
                trunc_masks = [anno["trunc_mask"] for anno in dataset_dict["annotations"]]
                fg_mask = sum(trunc_masks).astype("bool").astype("uint8")
            else:
                visib_masks = [anno["segmentation"] for anno in dataset_dict["annotations"]]
                fg_mask = sum(visib_masks).astype("bool").astype("uint8")

            if self.with_depth and self.with_bg_depth:
                image, bg_depth = self.replace_bg(
                    image.copy(),
                    fg_mask,
                    return_mask=False,
                    truncate_fg=False,
                    with_bg_depth=True,
                    depth_bp=(in_depth_channels == 3),
                )
            else:
                image = self.replace_bg(image.copy(), fg_mask, return_mask=False, truncate_fg=False)
        ######## replace bg done #############################################################################

        # NOTE: maybe add or change color augment here ===================================
        if cfg.INPUT.WITH_IMG and self.color_aug_prob > 0 and self.color_augmentor is not None:
            if np.random.rand() < self.color_aug_prob:
                if cfg.INPUT.COLOR_AUG_SYN_ONLY and img_type not in ["real"]:
                    image = self._color_aug(image, self.color_aug_type)
                else:
                    image = self._color_aug(image, self.color_aug_type)

        # other transforms (mainly geometric ones) ---------------------------------
        # for 6d pose task, flip is not allowed in general except for some 2d keypoints methods
        image, transforms = T.apply_augmentations(self.augmentation, image)
        im_H, im_W = image_shape = image.shape[:2]  # h, w

        if input_cfg.WITH_IMG:
            # NOTE: scale camera intrinsic if necessary ================================
            scale_x = im_W / im_W_ori
            scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
            if im_W != im_W_ori or im_H != im_H_ori:
                dataset_dict["cam"][0] *= scale_x
                dataset_dict["cam"][1] *= scale_y
                K = dataset_dict["cam"].numpy()

            # image (normalized)-----------------------------------------
            image = self.normalize_image(cfg, image.transpose(2, 0, 1))  # CHW
            # CHW, float32 tensor
            dataset_dict["image"] = torch.as_tensor(image.astype("float32")).contiguous()

        if self.with_depth:
            if do_replace_bg and self.with_bg_depth:
                mask_bg_depth = (fg_mask == 0).astype(np.bool)  # must be bool
                depth[mask_bg_depth] = bg_depth[mask_bg_depth]

            if self.aug_depth:  # randomly fill 0 points
                depth_0_idx = depth[:, :, -1] == 0
                depth[depth_0_idx] = np.random.normal(np.median(depth[depth_0_idx]), 0.1, depth[depth_0_idx].shape)
            if self.aug_depth and np.random.rand(1) < self.drop_depth_prob:  # drop 20% of depth values
                keep_mask = np.random.uniform(0, 1, size=depth.shape[:2])
                keep_mask = keep_mask > self.drop_depth_ratio
                depth = depth * keep_mask[:, :, None]

            # add gaussian noise
            if self.aug_depth and np.random.rand(1) < self.add_noise_depth_prob:
                # # add gaussian noise to >0 regions
                # depth_idx = depth > 0
                # depth[depth_idx] += np.random.normal(0, 0.01, depth[depth_idx].shape)
                depth = add_noise_depth(depth, level=self.add_noise_depth_level)

            # maybe need to resize
            if im_W != im_W_ori or im_H != im_H_ori:
                depth = mmcv.imresize(depth, (im_W, im_H, in_depth_channels), interpolation="nearest")

            dataset_dict["depth"] = torch.as_tensor(
                rearrange(depth.reshape(im_H, im_W, in_depth_channels), "h w c -> c h w")
            ).contiguous()

        ## for train ####################################################################################
        if "annotations" in dataset_dict:
            dset_meta = MetadataCatalog.get(dataset_name)
            # transform annotations ------------------------
            # NOTE: newly added keys should consider whether they need to be transformed
            annos = [
                transform_instance_annotations(obj_anno, transforms, image_shape)
                for obj_anno in dataset_dict.pop("annotations")
            ]

            # construct instances ===============================
            # obj_classes, obj_poses, obj_scales, obj_trunc_masks, mug_handle, inst_name
            instances = annotations_to_instances(cfg, annos, image_shape)

            # sample point_cloud and nocs ===============================
            pcl_list = []
            nocs_list = []
            rgb_list = []
            _trunc_fg = cfg.INPUT.get("TRUNCATE_FG", False)
            if _trunc_fg:
                masks = instances.obj_trunc_masks.tensor
            else:
                masks = instances.obj_visib_masks.tensor
            depth_bp = misc.backproject_th(torch.from_numpy(depth[:, :, -1]), K)
            image_th = torch.from_numpy(image)
            poses = instances.obj_poses.tensor
            scales = instances.obj_scales
            cat_ids = instances.obj_classes
            inst_names = instances.inst_names
            for mask, pose, scale, cat_id, inst_name in zip(masks, poses, scales, cat_ids, inst_names):
                if input_cfg.SAMPLE_DEPTH_FROM_BALL:
                    rgb, pcl, nocs = crop_ball_from_depth_image(
                        image_th,
                        depth_bp,
                        mask,
                        pose,
                        scale,
                        coord=coord,
                        ratio=input_cfg.DEPTH_SAMPLE_BALL_RATIO,
                        cam_intrinsics=torch.from_numpy(K),
                        num_points=input_cfg.NUM_PCL,
                        device="cpu",
                        fps_sample=input_cfg.FPS_SAMPLE,
                    )
                else:  # sample randomly in mask
                    rgb, pcl, nocs = crop_mask_depth_image(
                        image_th, depth_bp, mask, coord=coord, num_points=input_cfg.NUM_PCL
                    )

                rgb_list.append(rgb.to(torch.float32).contiguous())
                pcl_list.append(pcl.to(torch.float32).contiguous())

                # NOTE: mv the nocs map of mug because the incorrespondence of pose/model definition
                if dset_meta.objs[cat_id] == "mug":
                    t0 = self.mug_meta[inst_name][0].astype(np.float32)
                    s0 = self.mug_meta[inst_name][1].astype(np.float32)
                    nocs = s0 * (nocs + t0)  # nocs shape [num_p, 3]

                nocs_list.append(nocs.to(torch.float32).T.contiguous())  # [3, num_p]

            instances.rgb = torch.stack(rgb_list, dim=0)
            instances.pcl = torch.stack(pcl_list, dim=0)
            instances.nocs = torch.stack(nocs_list, dim=0)

            # obj_sym_infos, mean_model_points ===============================
            obj_sym_infos = []
            obj_mean_points = []
            obj_mean_scales = []
            if self.with_last_poses:
                last_frame_poses = []

            if input_cfg.KPS_TYPE.lower() == "fps":
                obj_fps_points = []

            for obj_id, mug_handle, inst_name in zip(
                instances.obj_classes, instances.mug_handles, instances.inst_names
            ):
                sym_infos, obj_name = self._get_sym_infos(dataset_name, obj_id, mug_handle)
                obj_sym_infos.append(sym_infos)
                obj_mean_scales.append(self.mean_scales[int(obj_id)])
                if "cmra" in dataset_name and input_cfg.USE_CMRA_MODEL:
                    ref_key = inst_name
                else:
                    ref_key = obj_name
                obj_mean_points.append(self.mean_model_dict[ref_key])
                if self.with_last_poses:
                    last_frame_poses.append(self.last_frame_pose_dict[scene_im_id][inst_name])
                if input_cfg.KPS_TYPE.lower() == "fps":
                    obj_fps_points.append(self._get_fps_points(dataset_name, inst_name))

            instances.obj_sym_infos = MyList(obj_sym_infos)
            instances.obj_mean_scales = torch.tensor(np.array(obj_mean_scales).astype("float32"))
            instances.obj_mean_points = torch.tensor(np.array(obj_mean_points).astype("float32"))

            if self.with_last_poses:
                instances.last_frame_poses = torch.tensor(np.array(last_frame_poses).astype("float32"))

            if input_cfg.KPS_TYPE.lower() == "fps":
                instances.obj_fps_points = torch.tensor(np.array(obj_fps_points).astype("float32"))

            instances.remove("inst_names")
            # instances infos ===============================
            # NOTE: if no instance left after filtering, will return un-filtered one
            dataset_dict["instances"] = filter_empty_instances(instances, by_box=False, by_mask=True)

        return dataset_dict

    def read_data_test(self, dataset_dict):
        """load image and annos random shift & scale bbox; crop, rescale."""
        assert self.split != "train", self.split
        cfg = self.cfg
        input_cfg = cfg.INPUT
        net_cfg = cfg.MODEL.CATRE
        backbone_cfg = net_cfg.BACKBONE
        pcl_net_cfg = net_cfg.PCLNET

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        dataset_name = dataset_dict["dataset_name"]

        # load mean scale  ================================
        self._get_mean_scale(dataset_name)

        file_name = dataset_dict["file_name"]
        image = read_image_mmcv(dataset_dict["file_name"], format=self.img_format)
        # should be consistent with the size in dataset_dict
        utils.check_image_size(dataset_dict, image)
        im_H_ori, im_W_ori = image.shape[:2]

        # for 6d pose task, flip is not allowed in general except for some 2d keypoints methods
        image, transforms = T.apply_augmentations(self.augmentation, image)
        im_H, im_W = image_shape = image.shape[:2]  # h, w

        # NOTE: scale camera intrinsic if necessary ================================
        scale_x = im_W / im_W_ori
        scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
        if "cam" in dataset_dict:
            if im_W != im_W_ori or im_H != im_H_ori:
                dataset_dict["cam"][0] *= scale_x
                dataset_dict["cam"][1] *= scale_y
            K = dataset_dict["cam"].astype("float32")
            dataset_dict["cam"] = torch.as_tensor(K)
        else:
            raise RuntimeError("cam intrinsic is missing")

        # image (normalized)-----------------------------------------
        image = self.normalize_image(cfg, image.transpose(2, 0, 1))
        # CHW, float32 tensor
        dataset_dict["image"] = torch.as_tensor(image.astype("float32")).contiguous()

        # CHW -> HWC
        # coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)

        ## load depth
        in_depth_channels = 0
        if self.with_depth:
            assert "depth_file" in dataset_dict, "depth file is not in dataset_dict"
            depth_path = dataset_dict["depth_file"]
            log_first_n(logging.WARN, "with depth", n=1)
            depth = load_depth(depth_path)  # uint16
            assert depth is not None
            depth = depth.astype("float32") / 1000.0  # to m

            if cfg.INPUT.BP_DEPTH:
                in_depth_channels = 3
                depth = misc.backproject(depth, K)
            else:
                in_depth_channels = 1
                depth = depth[:, :, None]  # hwc

            dataset_dict["depth"] = torch.as_tensor(
                rearrange(depth.reshape(im_H, im_W, in_depth_channels), "h w c -> c h w")
            ).contiguous()

        ## for test ############################################################################
        # determine test box and init pose type---------------------------------------------
        bbox_key, pose_key = get_test_bbox_initpose_key(cfg)
        # ---------------------------------------
        # "annotations" means detections
        test_annos = dataset_dict["annotations"]
        obj_sym_infos = []
        obj_mean_points = []
        obj_mean_scales = []
        if input_cfg.KPS_TYPE.lower() == "fps":
            obj_fps_points = []
        for inst_i, test_anno in enumerate(test_annos):
            obj_id = test_anno["category_id"]
            mug_handle = test_anno["mug_handle"]
            sym_infos, obj_name = self._get_sym_infos(dataset_name, obj_id, mug_handle)
            obj_sym_infos.append(sym_infos)
            if "cmra" in dataset_name and input_cfg.USE_CMRA_MODEL:
                ref_key = inst_name
            else:
                ref_key = obj_name
            obj_mean_points.append(self.mean_model_dict[ref_key])
            obj_mean_scales.append(self.mean_scales[obj_id])
            if input_cfg.KPS_TYPE.lower() == "fps":
                # NOTE: fps needs exact instance name
                inst_name = test_anno["inst_name"]
                obj_fps_points.append(self._get_fps_points(dataset_name, inst_name))

            # get test bbox, init pose(maybe None)  # keys become: bbox, [pose]
            self._get_test_bbox_initpose(
                test_anno, bbox_key, pose_key, K=K, dataset_name=dataset_name, imW=im_W, imH=im_H
            )

        test_annos = [
            transform_instance_annotations(_anno, transforms, image_shape) for _anno in dataset_dict.pop("annotations")
        ]
        # construct test instances ===============================
        # obj_classes, obj_boxes_det, [obj_poses]
        test_insts = annotations_to_instances(cfg, test_annos, image_shape)
        # sample point_cloud, NOTE: mask are estimated by dualpose net
        pcl_list = []
        rgb_list = []
        _trunc_fg = cfg.INPUT.get("TRUNCATE_FG", False)
        if _trunc_fg:
            masks = test_insts.obj_trunc_masks.tensor
        else:
            masks = test_insts.obj_visib_masks.tensor
        depth_bp = misc.backproject_th(torch.from_numpy(depth[:, :, -1]), K)
        image_th = torch.from_numpy(image)

        poses = test_insts.obj_poses.tensor
        scales = test_insts.obj_scales
        bboxes = test_insts.obj_boxes
        for mask, pose, scale, bbox in zip(masks, poses, scales, bboxes):
            # random occlude the object inner bbox
            if input_cfg.OCCLUDE_MASK_TEST:
                mask, occlude_ratio = occlude_obj_by_bboxes(bbox, mask)
            if input_cfg.SAMPLE_DEPTH_FROM_BALL:
                rgb, pcl, _ = crop_ball_from_depth_image(
                    image_th,
                    depth_bp,
                    mask,
                    pose,
                    scale,
                    ratio=input_cfg.DEPTH_SAMPLE_BALL_RATIO,
                    cam_intrinsics=torch.from_numpy(K),
                    num_points=input_cfg.NUM_PCL,
                    device="cpu",
                    fps_sample=input_cfg.FPS_SAMPLE,
                )
            else:  # sample randomly in mask
                rgb, pcl, _ = crop_mask_depth_image(image_th, depth_bp, mask, num_points=input_cfg.NUM_PCL)

            pcl_list.append(pcl.to(torch.float32).contiguous())
            rgb_list.append(rgb.to(torch.float32).contiguous())

        test_insts.pcl = torch.stack(pcl_list, dim=0)
        test_insts.rgb = torch.stack(rgb_list, dim=0)
        # obj_extents, obj_sym_infos, obj_fps_points(for ablation) --------------------------------------------
        test_insts.obj_sym_infos = MyList(obj_sym_infos)
        test_insts.obj_mean_points = torch.tensor(np.array(obj_mean_points).astype("float32"))
        test_insts.obj_mean_scales = torch.tensor(np.array(obj_mean_scales).astype("float32"))

        if input_cfg.KPS_TYPE.lower() == "fps":
            test_insts.obj_fps_points = torch.tensor(np.array(obj_fps_points).astype("float32"))
            test_insts.remove("inst_names")

        dataset_dict["instances"] = test_insts
        return dataset_dict

    def _get_test_bbox_initpose(self, test_anno, bbox_key, pose_key, K, dataset_name, imW=640, imH=480):
        cfg = self.cfg
        # get init poses (or None) -------------------------------------------------
        if pose_key == "pose_gt_noise":
            # in this case, we load the gt poses/annotations and add random noise
            gt_pose = test_anno["pose"]
            pose_gt_noise = aug_poses_normal_np(
                gt_pose[None],
                std_rot=cfg.INPUT.NOISE_ROT_STD_TEST,
                std_trans=cfg.INPUT.NOISE_TRANS_STD_TEST,
                max_rot=cfg.INPUT.NOISE_ROT_MAX_TEST,
                min_z=cfg.INPUT.INIT_TRANS_MIN_Z,
            )[0]
            gt_scale = test_anno["scale"]
            scale_gt_noise = aug_scale_normal_np(
                gt_scale[None],
                std_scale=cfg.INPUT.NOISE_SCALE_STD_TEST,
                min_s=cfg.INPUT.INIT_SCALE_MIN,
            )[0]
            test_anno["pose"] = pose_gt_noise
            test_anno["scale"] = scale_gt_noise
        elif pose_key == "pose_est":
            test_anno["pose"] = test_anno.pop("pose_est")
            test_anno["scale"] = test_anno.pop("scale_est")
        else:
            raise ValueError("Unknown test init_pose type: {}".format(pose_key))

        # get test boxes ------------------------------------------------------
        if bbox_key != "bbox_est" and "bbox_est" in test_anno:
            test_anno["bbox_det"] = test_anno["bbox_est"]
            test_anno["bbox_det_mode"] = test_anno["bbox_mode"]

        if bbox_key == "bbox_est":
            test_anno["bbox"] = test_anno.pop("bbox_est")
            test_anno["bbox_det"] = test_anno["bbox"]
            test_anno["bbox_det_mode"] = test_anno["bbox_mode"]

        elif bbox_key == "bbox_gt_aug":
            bbox_gt = BoxMode.convert(test_anno.pop("bbox"), test_anno["bbox_mode"], BoxMode.XYXY_ABS)
            test_anno["bbox"] = self.aug_bbox_non_square(cfg, bbox_gt, im_H=imH, im_W=imW)
            test_anno["bbox_mode"] = BoxMode.XYXY_ABS
        else:  # gt
            if "bbox" not in test_anno:
                raise RuntimeError("No gt bbox for test!")

        # inplace modification, do not return

    def __getitem__(self, idx):
        if self.split != "train":
            dataset_dict = self._get_sample_dict(idx)
            return self.read_data(dataset_dict)

        while True:  # return valid data for train
            dataset_dict = self._get_sample_dict(idx)
            processed_data = self.read_data(dataset_dict)
            if processed_data is None:
                idx = self._rand_another(idx)
                continue
            return processed_data


def build_catre_train_loader(cfg, dataset_names):
    """A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg: the config

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts = get_detection_dataset_dicts(
        dataset_names,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    dataset_dicts = filter_invalid_in_dataset_dicts(dataset_dicts, visib_thr=cfg.DATALOADER.FILTER_VISIB_THR)

    dataset = CATRE_DatasetFromList(cfg, split="train", lst=dataset_dicts, copy=False, flatten=False)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return my_build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_catre_test_loader(cfg, dataset_name, train_objs=None):
    """Similar to `build_detection_train_loader`. But this function uses the
    given `dataset_name` argument (instead of the names in cfg), and uses batch
    size 1.

    Args:
        cfg:
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # load test init bbox / masks / pose & scale ------------------------------------
    if cfg.MODEL.LOAD_POSES_TEST:
        init_pose_files = cfg.DATASETS.INIT_POSE_FILES_TEST
        assert len(init_pose_files) == len(cfg.DATASETS.TEST)
        load_catre_init_into_dataset(
            dataset_name,
            dataset_dicts,
            init_pose_file=init_pose_files[cfg.DATASETS.TEST.index(dataset_name)],
            score_thr=cfg.DATASETS.INIT_POSE_THR,
            train_objs=train_objs,
            with_masks=True,
            with_bboxes=True,
        )
        if cfg.DATALOADER.FILTER_EMPTY_DETS:
            dataset_dicts = filter_empty_dets(dataset_dicts)

    dataset = CATRE_DatasetFromList(cfg, split="test", lst=dataset_dicts, flatten=False)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    # Horovod: limit # of CPU threads to be used per worker.
    # if num_workers > 0:
    #     torch.set_num_threads(num_workers)

    kwargs = {"num_workers": num_workers}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    # https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
    # if (num_workers > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        **kwargs,
    )
    return data_loader


def get_test_bbox_initpose_key(cfg):
    # NOTE: we don't have bbox from pose_est in category-level estimation
    test_bbox_type = cfg.INPUT.BBOX_TYPE_TEST  # $est$ | gt
    test_pose_init_type = cfg.INPUT.INIT_POSE_TYPE_TEST  # gt_noise | $est$ | canonical
    bbox_initpose_types_to_keys = {
        "est/est": ("bbox_est", "pose_est"),  # common test case 1
        # these are only for validation
        "gt/gt_noise": ("bbox", "pose_gt_noise"),
    }
    return bbox_initpose_types_to_keys[f"{test_bbox_type}/{test_pose_init_type}"]
