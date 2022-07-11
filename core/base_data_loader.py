# -*- coding: utf-8 -*-
import copy
import hashlib
import logging
import os
import os.path as osp
import random
import cv2
import mmcv
import numpy as np
import pickle

from core.utils.augment import AugmentRGB
import torch.utils.data as data
from core.utils.dataset_utils import flat_dataset_dicts
from lib.utils.utils import lazy_property
from lib.utils.fs import recursive_walk
from core.utils.data_utils import resize_short_edge, read_image_mmcv
from lib.pysixd import misc
from lib.utils.config_utils import try_get_key


logger = logging.getLogger(__name__)


class Base_DatasetFromList(data.Dataset):
    """# https://github.com/facebookresearch/detectron2/blob/master/detectron2/
    data/common.py Wrap a list to a torch Dataset.

    It produces elements of the list as data.
    """

    def __init__(
        self,
        *,
        cfg,
        split,
        lst: list,
        copy: bool = True,
        serialize: bool = True,
        flatten=True,
    ):
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
        # fmt: off
        self.img_format = try_get_key(cfg, "INPUT.FORMAT", "img_format")  # default BGR
        self.with_depth = try_get_key(cfg, "INPUT.WITH_DEPTH", "with_depth")
        self.aug_depth = try_get_key(cfg, "INPUT.AUG_DEPTH", "aug_depth")
        # NOTE: color augmentation config
        self.color_aug_prob = try_get_key(cfg, "INPUT.COLOR_AUG_PROB", "color_aug_prob")
        self.color_aug_type = try_get_key(cfg, "INPUT.COLOR_AUG_TYPE", "color_aug_type")
        self.color_aug_code = try_get_key(cfg, "INPUT.COLOR_AUG_CODE", "color_aug_code")

        # fmt: on
        self.cfg = cfg
        self.split = split  # train | val | test
        if split == "train" and self.color_aug_prob > 0:
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
        else:
            self.color_augmentor = None
        self._lst = flat_dataset_dicts(lst) if flatten else lst
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

    def read_data(self, dataset_dict):
        if self.split == "train":
            return self.read_data_train(dataset_dict)
        else:
            return self.read_data_test(dataset_dict)

    def read_data_train(self, dataset_dict):
        assert self.split == "train", self.split
        raise NotImplementedError()

    def read_data_test(self, dataset_dict):
        assert self.split != "train", self.split
        raise NotImplementedError()

    def _rand_another(self, idx):
        pool = [i for i in range(self.__len__()) if i != idx]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        dataset_dict = self._get_sample_dict(idx)

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # NOTE: subclass need to re-implement this part
        return dataset_dict

    def _get_sample_dict(self, idx):
        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            dataset_dict = pickle.loads(bytes)
        elif self._copy:
            dataset_dict = copy.deepcopy(self._lst[idx])
        else:
            dataset_dict = self._lst[idx]
        return dataset_dict

    def normalize_image(self, cfg, image):
        """
        cfg: upper format, the whole cfg; lower format, the input_cfg
        image: CHW format
        """
        pixel_mean = np.array(try_get_key(cfg, "MODEL.PIXEL_MEAN", "pixel_mean")).reshape(-1, 1, 1)
        pixel_std = np.array(try_get_key(cfg, "MODEL.PIXEL_STD", "pixel_std")).reshape(-1, 1, 1)
        return (image - pixel_mean) / pixel_std

    def aug_bbox_non_square(self, cfg, bbox_xyxy, im_H, im_W):
        """Similar to DZI, but the resulted bbox is not square, and not enlarged
        Args:
            cfg: upper format, the whole cfg; lower format, the input_cfg
            bbox_xyxy (np.ndarray): (4,)
            im_H (int):
            im_W (int):
        Returns:
             augmented bbox (ndarray)
        """
        x1, y1, x2, y2 = bbox_xyxy.copy()
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bh = y2 - y1
        bw = x2 - x1
        bbox_aug_type = try_get_key(cfg, "INPUT.BBOX_AUG_TYPE", "bbox_aug_type").lower()
        if bbox_aug_type == "uniform":
            # different to DZI: scale both w and h
            scale_ratio = 1 + try_get_key(cfg, "INPUT.BBOX_AUG_SCALE_RATIO", "bbox_aug_scale_ratio") * (
                2 * np.random.random_sample(2) - 1
            )  # [1-0.25, 1+0.25]
            shift_ratio = try_get_key(cfg, "INPUT.BBOX_AUG_SHIFT_RATIO", "bbox_aug_shift_ratio") * (
                2 * np.random.random_sample(2) - 1
            )  # [-0.25, 0.25]
            bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
            new_bw = bw * scale_ratio[0]
            new_bh = bh * scale_ratio[1]
            x1 = min(max(bbox_center[0] - new_bw / 2, 0), im_W)
            y1 = min(max(bbox_center[1] - new_bh / 2, 0), im_W)
            x2 = min(max(bbox_center[0] + new_bw / 2, 0), im_W)
            y2 = min(max(bbox_center[1] + new_bh / 2, 0), im_W)
            bbox_auged = np.array([x1, y1, x2, y2])
        elif bbox_aug_type == "roi10d":
            # shift (x1,y1), (x2,y2) by 15% in each direction
            _a = -0.15
            _b = 0.15
            x1 += bw * (np.random.rand() * (_b - _a) + _a)
            x2 += bw * (np.random.rand() * (_b - _a) + _a)
            y1 += bh * (np.random.rand() * (_b - _a) + _a)
            y2 += bh * (np.random.rand() * (_b - _a) + _a)
            x1 = min(max(x1, 0), im_W)
            x2 = min(max(x1, 0), im_W)
            y1 = min(max(y1, 0), im_H)
            y2 = min(max(y2, 0), im_H)
            bbox_auged = np.array([x1, y1, x2, y2])
        elif bbox_aug_type == "truncnorm":
            raise NotImplementedError("BBOX_AUG_TYPE truncnorm is not implemented yet.")
        else:
            bbox_auged = bbox_xyxy.copy()
        return bbox_auged

    def aug_bbox_DZI(self, cfg, bbox_xyxy, im_H, im_W):
        """Used for DZI, the augmented box is a square (maybe enlarged)
        Args:
            cfg: upper format, the whole cfg; lower format, the input_cfg
            bbox_xyxy (np.ndarray):
        Returns:
             center, scale
        """
        x1, y1, x2, y2 = bbox_xyxy.copy()
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bh = y2 - y1
        bw = x2 - x1
        dzi_type = try_get_key(cfg, "INPUT.DZI_TYPE", "dzi_type").lower()
        dzi_pad_scale = try_get_key(cfg, "INPUT.DZI_PAD_SCALE", "dzi_pad_scale")
        if dzi_type == "uniform":
            dzi_scale_ratio = try_get_key(cfg, "INPUT.DZI_SCALE_RATIO", "dzi_scale_ratio")
            dzi_shift_ratio = try_get_key(cfg, "INPUT.DZI_SHIFT_RATIO", "dzi_shift_ratio")

            scale_ratio = 1 + dzi_scale_ratio * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
            shift_ratio = dzi_shift_ratio * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
            bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
            scale = max(y2 - y1, x2 - x1) * scale_ratio * dzi_pad_scale
        elif dzi_type == "roi10d":
            # shift (x1,y1), (x2,y2) by 15% in each direction
            _a = -0.15
            _b = 0.15
            x1 += bw * (np.random.rand() * (_b - _a) + _a)
            x2 += bw * (np.random.rand() * (_b - _a) + _a)
            y1 += bh * (np.random.rand() * (_b - _a) + _a)
            y2 += bh * (np.random.rand() * (_b - _a) + _a)
            x1 = min(max(x1, 0), im_W)
            x2 = min(max(x1, 0), im_W)
            y1 = min(max(y1, 0), im_H)
            y2 = min(max(y2, 0), im_H)
            bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
            scale = max(y2 - y1, x2 - x1) * dzi_pad_scale
        elif dzi_type == "truncnorm":
            raise NotImplementedError("DZI truncnorm not implemented yet.")
        else:
            bbox_center = np.array([cx, cy])  # (w/2, h/2)
            scale = max(y2 - y1, x2 - x1)
        scale = min(scale, max(im_H, im_W)) * 1.0
        return bbox_center, scale

    def _get_color_augmentor(self, aug_type="ROI10D", aug_code=None):
        # fmt: off
        if aug_type.lower() == "roi10d":
            color_augmentor = AugmentRGB(
                brightness_delta=2.5 / 255.,  # 0,
                lighting_std=0.3,
                saturation_var=(0.95, 1.05),  #(1, 1),
                contrast_var=(0.95, 1.05))  # (1, 1))  #
        elif aug_type.lower() == "aae":
            import imgaug.augmenters as iaa  # noqa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast)  # noqa
            aug_code = """Sequential([
                # Sometimes(0.5, PerspectiveTransform(0.05)),
                # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
                # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
                Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),
                Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
                Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
                Sometimes(0.3, Invert(0.2, per_channel=True)),
                Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
                Sometimes(0.5, Multiply((0.6, 1.4))),
                Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))
                ], random_order = False)"""
            # for darker objects, e.g. LM driller: use BOOTSTRAP_RATIO: 16 and weaker augmentation
            aug_code_weaker = """Sequential([
                Sometimes(0.4, CoarseDropout( p=0.1, size_percent=0.05) ),
                # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
                Sometimes(0.5, GaussianBlur(np.random.rand())),
                Sometimes(0.5, Add((-20, 20), per_channel=0.3)),
                Sometimes(0.4, Invert(0.20, per_channel=True)),
                Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),
                Sometimes(0.5, Multiply((0.7, 1.4))),
                Sometimes(0.5, LinearContrast((0.5, 2.0), per_channel=0.3))
                ], random_order=False)"""
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == "code":  # assume imgaug
            import imgaug.augmenters as iaa
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast)  # noqa
            aug_code = self.color_aug_code
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == 'code_albu':
            from albumentations import (HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
                                        Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion,
                                        HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
                                        MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast,
                                        RandomBrightness, Flip, OneOf, Compose, CoarseDropout, RGBShift, RandomGamma,
                                        RandomBrightnessContrast, JpegCompression, InvertImg)  # noqa
            aug_code = """Compose([
                CoarseDropout(max_height=0.05*480, max_holes=0.05*640, p=0.4),
                OneOf([
                    IAAAdditiveGaussianNoise(p=0.5),
                    GaussNoise(p=0.5),
                ], p=0.2),
                OneOf([
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                OneOf([
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(),
                ], p=0.3),
                InvertImg(p=0.2),
                RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=0.5),
                RandomContrast(limit=0.9, p=0.5),
                RandomGamma(gamma_limit=(80,120), p=0.5),
                RandomBrightness(limit=1.2, p=0.5),
                HueSaturationValue(hue_shift_limit=172, sat_shift_limit=20, val_shift_limit=27, p=0.3),
                JpegCompression(quality_lower=4, quality_upper=100, p=0.4),
            ], p=0.8)"""
            color_augmentor = eval(self.color_aug_code)
        else:
            color_augmentor = None
        # fmt: on
        return color_augmentor

    def _color_aug(self, image, aug_type="ROI10D"):
        # assume image in [0, 255] uint8
        if aug_type.lower() == "roi10d":  # need normalized image in [0,1]
            image = np.asarray(image / 255.0, dtype=np.float32).copy()
            image = self.color_augmentor.augment(image)
            image = (image * 255.0 + 0.5).astype(np.uint8)
            return image
        elif aug_type.lower() in ["aae", "code"]:
            # imgaug need uint8
            return self.color_augmentor.augment_image(image)
        elif aug_type.lower() in ["code_albu"]:
            augmented = self.color_augmentor(image=image)
            return augmented["image"]
        else:
            raise ValueError("aug_type: {} is not supported.".format(aug_type))

    @lazy_property
    def _bg_img_paths(self):
        logger.info("get bg image paths")
        cfg = self.cfg
        # random.choice(bg_img_paths)
        bg_type = try_get_key(cfg, "INPUT.BG_TYPE", "bg_type")
        bg_root = try_get_key(cfg, "INPUT.BG_IMGS_ROOT", "bg_imgs_root")
        num_bg_imgs = try_get_key(cfg, "INPUT.NUM_BG_IMGS", "num_bg_imgs")
        hashed_file_name = hashlib.md5(
            ("{}_{}_{}_get_bg_imgs".format(bg_root, num_bg_imgs, bg_type)).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(".cache/bg_paths_{}_{}.pkl".format(bg_type, hashed_file_name))
        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        if osp.exists(cache_path):
            logger.info("get bg_paths from cache file: {}".format(cache_path))
            bg_img_paths = mmcv.load(cache_path)
            logger.info("num bg imgs: {}".format(len(bg_img_paths)))
            assert len(bg_img_paths) > 0
            return bg_img_paths

        logger.info("building bg imgs cache {}...".format(bg_type))
        assert osp.exists(bg_root), f"BG ROOT: {bg_root} does not exist"
        if bg_type == "coco":
            img_paths = [
                osp.join(bg_root, fn.name) for fn in os.scandir(bg_root) if ".png" in fn.name or "jpg" in fn.name
            ]
        elif bg_type == "VOC_table":  # used in original deepim
            VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
            VOC_image_set_dir = osp.join(VOC_root, "ImageSets/Main")
            VOC_bg_list_path = osp.join(VOC_image_set_dir, "diningtable_trainval.txt")
            with open(VOC_bg_list_path, "r") as f:
                VOC_bg_list = [
                    line.strip("\r\n").split()[0] for line in f.readlines() if line.strip("\r\n").split()[1] == "1"
                ]
            img_paths = [osp.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx)) for bg_idx in VOC_bg_list]
        elif bg_type == "VOC":
            VOC_root = bg_root  # path to "VOCdevkit/VOC2012"
            img_paths = [
                osp.join(VOC_root, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(bg_root, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        elif bg_type == "SUN2012":
            img_paths = [
                osp.join(bg_root, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(bg_root, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        elif bg_type == "SUN_RGBD":
            assert try_get_key(cfg, "INPUT.WITH_BG_DEPTH", "with_bg_depth", default=False)
            img_paths = []
            for path in recursive_walk(bg_root):
                if "/depth/" in path:
                    depth_path = path
                    sample_dir = osp.join(osp.dirname(depth_path), "..")
                    im_names = os.listdir(osp.join(sample_dir, "image"))
                    im_path = osp.join(osp.join(sample_dir, "image"), im_names[0])

                    K_path = osp.join(sample_dir, "intrinsics.txt")
                    img_paths.append((im_path, depth_path, K_path))
        else:
            raise ValueError(f"BG_TYPE: {bg_type} is not supported")
        assert len(img_paths) > 0, len(img_paths)

        num_bg_imgs = min(len(img_paths), num_bg_imgs)
        indices = [i for i in range(len(img_paths))]
        sel_indices = np.random.choice(indices, num_bg_imgs)
        bg_img_paths = [img_paths[idx] for idx in sel_indices]

        mmcv.dump(bg_img_paths, cache_path)
        logger.info("num bg imgs: {}".format(len(bg_img_paths)))
        assert len(bg_img_paths) > 0
        return bg_img_paths

    def replace_bg(self, im, im_mask, return_mask=False, truncate_fg=False, with_bg_depth=False, depth_bp=False):
        cfg = self.cfg
        # add background to the image
        H, W = im.shape[:2]
        ind = random.randint(0, len(self._bg_img_paths) - 1)
        if with_bg_depth:
            filename, depth_path, K_path = self._bg_img_paths[ind]
            depth_factor = try_get_key(cfg, "INPUT.BG_DEPTH_FACTOR", "bg_depth_factor")
        else:
            filename = self._bg_img_paths[ind]
            depth_path = None
            K_path = None
            depth_factor = None

        if try_get_key(cfg, "INPUT.BG_KEEP_ASPECT_RATIO", "bg_keep_aspect_ratio", default=True):
            bg_img, bg_depth = self.get_bg_image(
                filename, H, W, depth_path=depth_path, depth_factor=depth_factor, bp_depth=depth_bp, K_path=K_path
            )
        else:
            bg_img, bg_depth = self.get_bg_image_v2(
                filename, H, W, depth_path=depth_path, depth_factor=depth_factor, bp_depth=depth_bp, K_path=K_path
            )

        if len(bg_img.shape) != 3:
            bg_img = np.zeros((H, W, 3), dtype=np.uint8)
            logger.warning("bad background image: {}".format(filename))

        mask = im_mask.copy().astype(np.bool)
        if truncate_fg:
            mask = self.trunc_mask(im_mask)
        mask_bg = ~mask
        im[mask_bg] = bg_img[mask_bg]
        im = im.astype(np.uint8)

        rets = [im]
        if with_bg_depth:
            rets.append(bg_depth)
        if return_mask:  # bool fg mask
            rets.append(mask)
        return tuple(rets) if len(rets) > 1 else rets[0]

    def trunc_mask(self, mask):
        # return the bool truncated mask
        mask = mask.copy().astype(np.bool)
        nonzeros = np.nonzero(mask.astype(np.uint8))
        x1, y1 = np.min(nonzeros, axis=1)
        x2, y2 = np.max(nonzeros, axis=1)
        c_h = 0.5 * (x1 + x2)
        c_w = 0.5 * (y1 + y2)
        rnd = random.random()
        # print(x1, x2, y1, y2, c_h, c_w, rnd, mask.shape)
        if rnd < 0.2:  # block upper
            c_h_ = int(random.uniform(x1, c_h))
            mask[:c_h_, :] = False
        elif rnd < 0.4:  # block bottom
            c_h_ = int(random.uniform(c_h, x2))
            mask[c_h_:, :] = False
        elif rnd < 0.6:  # block left
            c_w_ = int(random.uniform(y1, c_w))
            mask[:, :c_w_] = False
        elif rnd < 0.8:  # block right
            c_w_ = int(random.uniform(c_w, y2))
            mask[:, c_w_:] = False
        else:
            pass
        return mask

    def get_bg_image(
        self, filename, imH, imW, channel=3, depth_path=None, depth_factor=10000.0, bp_depth=False, K_path=None
    ):
        """keep aspect ratio of bg during resize target image size:

        imHximWxchannel.
        """
        target_size = min(imH, imW)
        max_size = max(imH, imW)
        real_hw_ratio = float(imH) / float(imW)
        bg_image = read_image_mmcv(filename, format=self.img_format)
        bg_h, bg_w, bg_c = bg_image.shape
        bg_image_resize = np.zeros((imH, imW, channel), dtype="uint8")
        if depth_path is not None:
            bg_depth = mmcv.imread(depth_path, "unchanged") / depth_factor
            depth_ch = 1
            if bp_depth:
                assert K_path is not None
                K = np.loadtxt(K_path).reshape(3, 3)
                bg_depth = misc.backproject(bg_depth, K)
                depth_ch = 3
            bg_depth_resize = np.zeros((imH, imW, depth_ch), dtype="float32")

        if (float(imH) / float(imW) < 1 and float(bg_h) / float(bg_w) < 1) or (
            float(imH) / float(imW) >= 1 and float(bg_h) / float(bg_w) >= 1
        ):
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                if bg_h_new < bg_h:
                    bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                    if depth_path is not None:
                        bg_depth_crop = bg_depth[0:bg_h_new, 0:bg_w]
                else:
                    bg_image_crop = bg_image
                    if depth_path is not None:
                        bg_depth_crop = bg_depth
            else:
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                if bg_w_new < bg_w:
                    bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
                    if depth_path is not None:
                        bg_depth_crop = bg_depth[0:bg_h, 0:bg_w_new]
                else:
                    bg_image_crop = bg_image
                    if depth_path is not None:
                        bg_depth_crop = bg_depth
        else:
            if bg_h >= bg_w:
                bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
                if depth_path is not None:
                    bg_depth_crop = bg_depth[0:bg_h_new, 0:bg_w]
            else:  # bg_h < bg_w
                bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
                # logger.info(bg_w_new)
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
                if depth_path is not None:
                    bg_depth_crop = bg_depth[0:bg_h, 0:bg_w_new]

        bg_image_resize_0 = resize_short_edge(bg_image_crop, target_size, max_size)
        h, w, c = bg_image_resize_0.shape
        bg_image_resize[0:h, 0:w, :] = bg_image_resize_0

        if depth_path is not None:
            bg_depth_resize_0 = resize_short_edge(bg_depth_crop, target_size, max_size, interpolation=cv2.INTER_NEAREST)
            if depth_ch == 1:
                bg_depth_resize_0 = bg_depth_resize_0[:, :, None]
            bg_depth_resize[0:h, 0:w, :] = bg_depth_resize_0
            return bg_image_resize, bg_depth_resize
        return bg_image_resize, None

    def get_bg_image_v2(
        self, filename, imH, imW, channel=3, depth_path=None, depth_factor=10000.0, bp_depth=False, K_path=None
    ):
        _bg_img = read_image_mmcv(filename, format=self.img_format)
        # randomly crop a region as background
        bw = _bg_img.shape[1]
        bh = _bg_img.shape[0]
        x1 = np.random.randint(0, int(bw / 3))
        y1 = np.random.randint(0, int(bh / 3))
        x2 = np.random.randint(int(2 * bw / 3), bw)
        y2 = np.random.randint(int(2 * bh / 3), bh)
        bg_img = cv2.resize(_bg_img[y1:y2, x1:x2], (imW, imH), interpolation=cv2.INTER_LINEAR)
        if depth_path is not None:
            _bg_depth = mmcv.imread(depth_path, "unchanged") / depth_factor
            depth_ch = 1
            if bp_depth:
                assert K_path is not None
                K = np.loadtxt(K_path).reshape(3, 3)
                _bg_depth = misc.backproject(_bg_depth, K)
                depth_ch = 3
            bg_depth = cv2.resize(
                _bg_depth[y1:y2, x1:x2], (imW, imH), interpolation=cv2.INTER_NEAREST
            )  # will drop last channel if it is HxWx1
            if depth_ch == 1:
                bg_depth = bg_depth[:, :, None]
            return bg_img, bg_depth
        return bg_img, None
