# encoding: utf-8
"""This file includes necessary params, info."""
import os
import mmcv
import os.path as osp

import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
# directory storing experiment data (result, model checkpoints, etc).
output_dir = osp.join(root_dir, "output")

data_root = osp.join(root_dir, "datasets")

# ---------------------------------------------------------------- #
# NOCS DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(data_root, "NOCS/")
real_train_dir = osp.join(dataset_root, "REAL/")

model_dir = osp.join(dataset_root, "obj_models")
mean_model_path = osp.join(model_dir, "normed_mean_points_emb_spd.npy")
cr_mean_model_path = osp.join(model_dir, "cr_normed_mean_model_points_spd.pkl")
train_model_path = osp.join(model_dir, "real_train_spd.pkl")
test_model_path = osp.join(model_dir, "real_test_spd.pkl")
scale_path = osp.join(model_dir, "abs_scale.pkl")

# object info
objects = ["bottle", "bowl", "camera", "can", "laptop", "mug"]

inst2obj = {
    # test insts
    'bottle_red_stanford_norm': "bottle",
    'bottle_shampoo_norm': "bottle",
    'bottle_shengjun_norm': "bottle",
    'bowl_blue_white_chinese_norm': "bowl",
    'bowl_shengjun_norm': "bowl",
    'bowl_white_small_norm': "bowl",
    'camera_canon_len_norm': "camera",
    'camera_canon_wo_len_norm': "camera",
    'camera_shengjun_norm': "camera",
    'can_arizona_tea_norm': "can",
    'can_green_norm': "can",
    'can_lotte_milk_norm': "can",
    'laptop_air_xin_norm': "laptop",
    'laptop_alienware_norm': "laptop",
    'laptop_mac_pro_norm': "laptop",
    'mug_anastasia_norm': "mug",
    'mug_brown_starbucks_norm': "mug",
    'mug_daniel_norm': "mug", 
    # train insts
    'bottle3_scene5_norm': "bottle",
    'bottle_blue_google_norm': "bottle",
    'bottle_starbuck_norm': "bottle",
    'bowl_blue_ikea_norm': "bowl",
    'bowl_brown_ikea_norm': "bowl",
    'bowl_chinese_blue_norm': "bowl",
    'camera_anastasia_norm': "camera",
    'camera_dslr_len_norm': "camera",
    'camera_dslr_wo_len_norm': "camera",
    'can_milk_wangwang_norm': "can",
    'can_porridge_norm': "can",
    'can_tall_yellow_norm': "can",
    'laptop_air_0_norm': "laptop",
    'laptop_air_1_norm': "laptop",
    'laptop_dell_norm': "laptop",
    'mug2_scene3_norm': "mug",
    'mug_vignesh_norm': "mug",
    'mug_white_green_norm': "mug",
}

insts = list(inst2obj.keys())

# only test
obj2inst = {
    "bottle": ['bottle_red_stanford_norm', 'bottle_shampoo_norm', 'bottle_shengjun_norm'], 
    "bowl": ['bowl_blue_white_chinese_norm', 'bowl_shengjun_norm', 'bowl_white_small_norm'], 
    "camera": ['camera_canon_len_norm', 'camera_canon_wo_len_norm', 'camera_shengjun_norm'], 
    "can": ['can_arizona_tea_norm', 'can_green_norm', 'can_lotte_milk_norm'], 
    "laptop":['laptop_air_xin_norm', 'laptop_alienware_norm', 'laptop_mac_pro_norm'], 
    "mug": ['mug_anastasia_norm', 'mug_brown_starbucks_norm', 'mug_daniel_norm'],
}

obj2id = {"bottle": 1, "bowl": 2, "camera": 3, "can": 4, "laptop": 5, "mug": 6}

obj_num = len(objects)

id2obj = {_id: _name for _name, _id in obj2id.items()}

# id2obj_camera = {1: "02876657", 2: "02880940", 3: "02942699", 4: "02946921", 5: "03642806", 6: "03797390"}
# objects_camera = list(id2obj_camera.values())
# obj2id_camera =  {_name: _id for _id, _name in id2obj_camera.items()}

# Camera info
width = 640
height = 480
center = (height / 2, width / 2)

intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float32)

mean_scale = {
    "bottle": 0.001 * np.array([87, 220, 89], dtype=np.float32),
    "bowl": 0.001 * np.array([165, 80, 165], dtype=np.float32),
    "camera": 0.001 * np.array([88, 128, 156], dtype=np.float32),
    "can": 0.001 * np.array([68, 146, 72], dtype=np.float32),
    "laptop": 0.001 * np.array([346, 200, 335], dtype=np.float32),
    "mug": 0.001 * np.array([146, 83, 114], dtype=np.float32),
}


def get_mean_bbox3d():
    mean_bboxes = {}
    for key, value in mean_scale.items():
        minx, maxx = -value[0] / 2, value[0] / 2
        miny, maxy = -value[1] / 2, value[1] / 2
        minz, maxz = -value[2] / 2, value[2] / 2

        mean_bboxes[key] = np.array(
            [
                [maxx, maxy, maxz],
                [minx, maxy, maxz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, maxy, minz],
                [minx, maxy, minz],
                [minx, miny, minz],
                [maxx, miny, minz],
            ],
            dtype=np.float32,
        )
    return mean_bboxes


def get_sym_info(obj_name, mug_handle=1):
    #  Y axis points upwards, x axis pass through the handle, z axis otherwise
    # return sym axis
    if obj_name == "bottle":
        sym = np.array([0, 1, 0], dtype=np.int)
    elif obj_name == "bowl":
        sym = np.array([0, 1, 0], dtype=np.int)
    elif obj_name == "camera":
        sym = None
    elif obj_name == "can":
        sym = np.array([0, 1, 0], dtype=np.int)
    elif obj_name == "laptop":
        sym = None
    elif obj_name == "mug":
        if mug_handle == 1:
            sym = None
        else:
            sym = np.array([0, 1, 0], dtype=np.int)
    else:
        raise NotImplementedError(f"No such a object class {obj_name}")
    return sym


def get_fps_points():
    """key is inst_name generated by
    core/catre/tools/nocs/nocs_fps_sample.py."""
    fps_points_path = osp.join(model_dir, "fps_points_spd.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict


def get_scales_dict():
    scales = mmcv.load(scale_path)
    return scales