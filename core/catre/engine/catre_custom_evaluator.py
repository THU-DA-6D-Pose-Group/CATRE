# -*- coding: utf-8 -*-
"""inference on dataset; save results; evaluate with custom evaluation
funcs."""
from email.mime import image
import itertools
import logging
import os.path as osp
import random
import time
from collections import OrderedDict
from tqdm import tqdm

import cv2
from tabulate import tabulate
import mmcv
import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator

cur_dir = osp.dirname(osp.abspath(__file__))
import ref
from core.utils.my_comm import all_gather, is_main_process, synchronize
from core.utils.my_visualizer import MyVisualizer, _GREEN, _GREY
from lib.pysixd import inout, misc
from lib.vis_utils.image import grid_show, vis_image_bboxes_cv2

from core.catre.engine.test_utils import compute_independent_mAP, bbox_xyxy_to_yxyx, pose_3x4_to_4x4

PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))


class CATRE_EvaluatorCustom(DatasetEvaluator):
    """custom evaluation of 6d pose.

    Assume single instance!!!
    """

    def __init__(
        self,
        cfg,
        dataset_name,
        distributed,
        output_dir,
        train_objs=None,
    ):
        self.cfg = cfg
        self.n_iter_test = cfg.MODEL.CATRE.N_ITER_TEST

        self._distributed = distributed
        self._output_dir = output_dir
        mmcv.mkdir_or_exist(output_dir)

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # if test objs are just a subset of train objs
        self.train_objs = train_objs

        self.dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(dataset_name)
        self.data_ref = ref.__dict__[self._metadata.ref_key]
        self.obj_names = self._metadata.objs
        self.obj_ids = [self.data_ref.obj2id[obj_name] for obj_name in self.obj_names]

        self._empty_pred = {
            "pred_class_ids": np.array([]).astype(np.int32),
            "pred_scores": np.array([]).astype(np.float32),
            "pred_bboxes": np.empty((0, 4), dtype=np.int32),
            "pred_RTs": np.empty((0, 4, 4), dtype=np.float32),
            "pred_scales": np.empty((0, 3), dtype=np.float32),
        }
        # eval cached
        self.use_cache = False
        if cfg.VAL.EVAL_CACHED or cfg.VAL.EVAL_PRINT_ONLY:
            self.use_cache = True
            for refine_i in range(self.n_iter_test + 1):
                self._eval_predictions(refine_i)  # mAP
            exit(0)

    def get_gts(self):
        self.gt_dict = OrderedDict()
        dataset_dicts = DatasetCatalog.get(self.dataset_name)
        self._logger.info("load gts of {}".format(self.dataset_name))
        for im_dict in tqdm(dataset_dicts):
            scene_im_id = im_dict["scene_im_id"]
            annos = im_dict["annotations"]
            image_path = im_dict["file_name"]
            gt_dict = dict(
                gt_class_ids=np.array([anno["category_id"] + 1 for anno in annos]),  # NOTE: start from 1
                gt_bboxes=np.array([bbox_xyxy_to_yxyx(anno["bbox"]) for anno in annos]),
                gt_RTs=np.array([pose_3x4_to_4x4(anno["pose"]) for anno in annos]),
                gt_scales=np.array([anno["scale"] for anno in annos]),
                gt_handle_visibility=np.array([anno["mug_handle"] for anno in annos]),
            )
            if scene_im_id not in self.gt_dict:
                self.gt_dict[scene_im_id] = gt_dict
                self.gt_dict[scene_im_id]["image_path"] = [image_path]
            else:
                self.gt_dict[scene_im_id]["image_path"].append(image_path)
                for k, v in gt_dict.items():
                    self.gt_dict[scene_im_id][k] = np.concatenate((self.gt_dict[scene_im_id][k], v), axis=0)

    def reset(self):
        self._predictions = []
        # when evaluate, transform the list to dict
        self._predictions_dict = OrderedDict()

    def _maybe_adapt_label_cls_name(self, label):
        """convert label in test dataset to label in train dataset; if not in
        train, return None."""
        if self.train_objs is not None:
            cls_name = self.obj_names[label]
            if cls_name not in self.train_objs:
                return None, None  # this class was not trained
            label = self.train_objs.index(cls_name)
        else:
            cls_name = self.obj_names[label]
        return label, cls_name

    def process(self, inputs, batch, outputs, out_dict):
        """
        Args:
            inputs: the batch from data loader
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "scene_im_id".
            batch: the batch (maybe flattened) to the model
            outputs:
            out_dict: the predictions of the model
        """
        pose_est_dict = {}
        scale_est_dict = {}
        for _i in range(self.n_iter_test + 1):
            pose_est_dict[f"iter{_i}"] = out_dict[f"pose_{_i}"].detach().cpu().numpy()
            scale_est_dict[f"iter{_i}"] = out_dict[f"scale_{_i}"].detach().cpu().numpy()

        batch_im_ids = batch["im_id"].detach().cpu().numpy().tolist()
        batch_inst_ids = batch["inst_id"].detach().cpu().numpy().tolist()
        batch_labels = batch["obj_cls"].detach().cpu().numpy().tolist()  # 0-based label into train set

        for im_i, (_input, output) in enumerate(zip(inputs, outputs)):
            for out_i, batch_im_i in enumerate(batch_im_ids):
                if im_i != int(batch_im_i):
                    continue
                scene_im_id = _input["scene_im_id"]

                if "instances" in _input:
                    inst_id = int(batch_inst_ids[out_i])
                    bbox = _input["instances"].obj_boxes.tensor[inst_id]
                    bbox_yxyx = bbox_xyxy_to_yxyx(bbox)
                    if _input["instances"].has("obj_scores"):
                        score = _input["instances"].obj_scores[inst_id]
                    else:
                        score = 1.0

                cur_label = batch_labels[out_i]
                pred_class_id = cur_label + 1  # NOTE: start from 1

                # get pose
                cur_result = {}
                for refine_i in range(self.n_iter_test + 1):
                    # NOTE: predict scale in real world and standard R matrix
                    scale_est = scale_est_dict[f"iter{refine_i}"][out_i]

                    pose_est = pose_est_dict[f"iter{refine_i}"][out_i]
                    pose_est = pose_3x4_to_4x4(pose_est)

                    cur_result = {
                        "pred_RTs": pose_est,
                        "pred_scales": scale_est,
                        "pred_class_ids": pred_class_id,
                        "pred_scores": score,
                        "pred_bboxes": bbox_yxyx,
                    }

                    self._predictions.append((scene_im_id, refine_i, cur_result))

    def batch_prediction_results(self, pred_list):
        pred = {}
        for key in pred_list[0].keys():
            pred[key] = np.array([p[key] for p in pred_list])
        return pred

    def _preds_list_to_dict(self):
        # list of tuple to dict
        for refine_i in range(self.n_iter_test + 1):
            self._predictions_dict[f"iter{refine_i}"] = {}

        for scene_im_id, refine_i, cur_result in self._predictions:
            if scene_im_id not in self._predictions_dict[f"iter{refine_i}"]:
                self._predictions_dict[f"iter{refine_i}"][scene_im_id] = []
            self._predictions_dict[f"iter{refine_i}"][scene_im_id].append(cur_result)

        for refine_i in range(self.n_iter_test + 1):
            for scene_im_id in self._predictions_dict[f"iter{refine_i}"]:
                self._predictions_dict[f"iter{refine_i}"][scene_im_id] = self.batch_prediction_results(
                    self._predictions_dict[f"iter{refine_i}"][scene_im_id]
                )  #

    def evaluate(self):
        if self._distributed:
            synchronize()
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))

            if not is_main_process():
                return

        self._preds_list_to_dict()
        eval_res = {}
        for refine_i in range(self.n_iter_test + 1):
            eval_res.update(self._eval_predictions(refine_i))  # recall
        return eval_res
        # return copy.deepcopy(self._eval_predictions())

    def _eval_predictions(self, cur_iter=0):
        """Evaluate self._predictions on 6d pose.

        Return results with the metrics of the tasks.
        """
        self._logger.info(f"Eval recalls of results at iter={cur_iter}...")

        cfg = self.cfg
        method_name = f"{cfg.EXP_ID.replace('_', '-')}"
        cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_preds.pkl")
        if cur_iter == 0:  # only load or dump results at iter0
            if osp.exists(cache_path) and self.use_cache:
                self._logger.info("load cached predictions")
                self._predictions_dict = mmcv.load(cache_path)
            else:
                if hasattr(self, "_predictions_dict"):
                    mmcv.dump(self._predictions_dict, cache_path)
                else:
                    raise RuntimeError("Please run inference first")
            self.get_gts()  # get self.gt_dict

        preds = self._predictions_dict[f"iter{cur_iter}"]

        pred_gt_merge_list = []
        for scene_im_id, gt in self.gt_dict.items():
            if scene_im_id in preds:
                gt.update(preds[scene_im_id])
            else:
                gt.update(self._empty_pred)
            pred_gt_merge_list.append(gt)

        synset_names = ["BG"] + self.obj_names
        degree_thresholds = [5, 10]
        shift_thresholds = [2, 5, 10]
        degree_shift_thresholds = [(5, 2), (5, 5), (10, 2), (10, 5), (10, 10)]
        iou_3d_thresholds = [0.1, 0.25, 0.50, 0.75]

        # average
        iou_3d_aps, pose_aps = compute_independent_mAP(
            pred_gt_merge_list,
            synset_names,
            degree_thresholds=degree_thresholds,
            shift_thresholds=shift_thresholds,
            iou_3d_thresholds=iou_3d_thresholds,
        )

        iou_metric_names = ["IoU25", "IoU50", "IoU75"]

        header = ["objects"] + self.obj_names + [f"Avg({len(self.obj_names)})"]
        big_tab = [header]

        for metric, thres in zip(iou_metric_names, iou_3d_thresholds[1:]):
            line = [metric]
            for idx, obj_name in enumerate(synset_names):
                if obj_name in self.obj_names:
                    line.append(f"{100*iou_3d_aps[idx, iou_3d_thresholds.index(thres)]:.2f}")
            # average
            line.append(f"{100*iou_3d_aps[-1, iou_3d_thresholds.index(thres)]:.2f}")
            big_tab.append(line)

        pose_metric_names = ["re5te2", "re5te5", "re10te2", "re10te5", "re10te10"]
        for metric, thres in zip(pose_metric_names, degree_shift_thresholds):
            line = [metric]
            degree_thre, shift_thre = thres
            for idx, obj_name in enumerate(synset_names):
                if obj_name in self.obj_names:
                    line.append(
                        f"{100*pose_aps[idx, degree_thresholds.index(degree_thre), shift_thresholds.index(shift_thre)]:.2f}"
                    )
            # average
            line.append(
                f"{100*pose_aps[-1, degree_thresholds.index(degree_thre), shift_thresholds.index(shift_thre)]:.2f}"
            )
            big_tab.append(line)

        re_metric_names = ["re5", "re10"]
        for metric, thres in zip(re_metric_names, degree_thresholds):
            line = [metric]
            degree_thre = thres
            for idx, obj_name in enumerate(synset_names):
                if obj_name in self.obj_names:
                    line.append(f"{100*pose_aps[idx, degree_thresholds.index(degree_thre), -1]:.2f}")
            # average
            line.append(f"{100*pose_aps[-1, degree_thresholds.index(degree_thre), -1]:.2f}")
            big_tab.append(line)

        te_metric_names = ["te2", "te5"]
        for metric, thres in zip(te_metric_names, shift_thresholds):
            line = [metric]
            shift_thre = thres
            for idx, obj_name in enumerate(synset_names):
                if obj_name in self.obj_names:
                    line.append(f"{100*pose_aps[idx, -1, shift_thresholds.index(shift_thre)]:.2f}")
            # average
            line.append(f"{100*pose_aps[-1, -1, shift_thresholds.index(shift_thre)]:.2f}")
            big_tab.append(line)

        res_log_tab_str = tabulate(
            big_tab,
            tablefmt="plain",
        )

        self._logger.info("\n{}".format(res_log_tab_str))

        dump_tab_name = osp.join(
            self._output_dir,
            f"{method_name}_{self.dataset_name}_tab_iter{cur_iter}.txt",
        )
        with open(dump_tab_name, "w") as f:
            f.write("{}\n".format(res_log_tab_str))

        if self._distributed:
            self._logger.warning("\n The current evaluation on multi-gpu is not correct, run with single-gpu instead.")

        return {}
