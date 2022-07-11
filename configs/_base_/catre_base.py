_base_ = "./common_base.py"

# -----------------------------------------------------------------------------
# Input (override common base)
# -----------------------------------------------------------------------------
INPUT = dict(
    # Whether the model needs RGB, YUV, HSV etc.
    FORMAT="BGR",
    MIN_SIZE_TRAIN=(480,),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TRAIN_SAMPLING="choice",
    MIN_SIZE_TEST=480,
    MAX_SIZE_TEST=640,
    WITH_DEPTH=True,
    BP_DEPTH=False,
    AUG_DEPTH=False,
    ALIGN_PCL=False,
    WITH_IMG=False,
    PCL_WITH_COLOR=False,
    SAMPLE_DEPTH_FROM_BALL=True,
    DEPTH_SAMPLE_BALL_RATIO=0.5,  # control the radius of sample depth ball
    FPS_SAMPLE=False,  # whether sample depth points using FPS
    # for symmtries
    MAX_SYM_DISC_STEP=0.01,
    # color aug
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE="ROI10D",
    COLOR_AUG_CODE="",
    COLOR_AUG_SYN_ONLY=False,
    ## bg images
    BG_TYPE="VOC_table",  # VOC_table | coco | VOC | SUN2012
    BG_IMGS_ROOT="datasets/VOCdevkit/VOC2012/",  # "datasets/coco/train2017/"
    NUM_BG_IMGS=10000,
    CHANGE_BG_PROB=0.0,  # prob to change bg of real image
    # truncation fg (randomly replace some side of fg with bg during replace_bg)
    TRUNCATE_FG=False,
    BG_KEEP_ASPECT_RATIO=True,
    ## input bbox type -------------------------------
    BBOX_TYPE_TEST="est",  #  est | gt | gt_aug (TODO)
    ## initial pose type ----------------------------------
    # gt_noise: using random perturbated pose based on gt
    # est: using external pose estimates
    # canonical: using DeepIM for initial pose prediction (like cosypose coarse+refine)
    # INIT_POSE_TYPE_TRAIN=["gt_noise"],
    INIT_POSE_TYPE_TRAIN=["gt_noise"],  # randomly chosen from ["gt_noise", "random", "canonical", "last_frame"]
    INIT_SCALE_TYPE_TRAIN=["gt_noise"],
    INIT_POSE_TRAIN_PATH="datasets/NOCS/train_init_poses/init_with_last_frame.pkl",
    INIT_POSE_TYPE_TEST="est",  # gt_noise | est | canonical
    ## pose init  (gt+noise) ----------------------------------
    NOISE_ROT_STD_TRAIN=(15, 10, 5, 2.5),  # randomly choose one
    NOISE_ROT_STD_TEST=15,  # if use gt pose + noise as random initial poses
    NOISE_ROT_MAX_TRAIN=45,
    NOISE_ROT_MAX_TEST=45,
    # trans
    NOISE_TRANS_STD_TRAIN=[(0.01, 0.01, 0.005), (0.01, 0.01, 0.01), (0.005, 0.005, 0.01)],
    NOISE_TRANS_STD_TEST=[(0.01, 0.01, 0.005), (0.01, 0.01, 0.01), (0.005, 0.005, 0.01)],
    INIT_TRANS_MIN_Z=0.1,
    # scale
    NOISE_SCALE_STD_TRAIN=[(0.001, 0.005, 0.001), (0.005, 0.001, 0.005), (0.01, 0.01, 0.01)],
    NOISE_SCALE_STD_TEST=[(0.001, 0.005, 0.001), (0.005, 0.001, 0.005), (0.01, 0.01, 0.01)],
    INIT_SCALE_MIN=0.04,
    ## pose init (random) -------------------------------------
    RANDOM_TRANS_MIN=[-0.35, -0.35, 0.5],
    RANDOM_TRANS_MAX=[0.35, 0.35, 1.3],
    RANDOM_SCALE_MIN=[0.04, 0.04, 0.04],
    RANDOM_SCALE_MAX=[0.5, 0.3, 0.4],
    # for refine ----------------------------------------------
    MEAN_MODEL_PATH="datasets/NOCS/obj_models/cr_normed_mean_model_points_spd.pkl",
    KPS_TYPE="bbox",  # bbox_from_scale | skeleton
    USE_CMRA_MODEL=True,
    WITH_NEG_AXIS=False,
    # train augmentation
    BBOX3D_AUG_PROB=0.0,
    RT_AUG_PROB=0.0,
    # TODO: test depth aug
    DEPTH_BILATERAL_FILTER_TEST=False,
    DEPTH_HOLE_FILL_TEST=False,
    NUM_KPS=32,
    NUM_PCL=1500,
    ZERO_CENTER_INPUT=False,
    NOCS_DIST_THER=0.03,
    ## pose init (canonical) ----------------------------------
    # a chain of axis-angle, the last value will be multiplied by pi
    CANONICAL_ROT=[(1, 0, 0, 0.5), (0, 0, 1, -0.7)],
    CANONICAL_TRANS=[0, 0, 1.0],
    CANONICAL_SIZE=[0.2, 0.2, 0.2],
    OCCLUDE_MASK_TEST=False,
)

# -----------------------------------------------------------------------------
# base model cfg for catre
# -----------------------------------------------------------------------------
MODEL = dict(
    DEVICE="cuda",
    WEIGHTS="",
    PIXEL_MEAN=[0, 0, 0],  # to [0,1]
    PIXEL_STD=[255.0, 255.0, 255.0],
    LOAD_POSES_TEST=False,
    REFINE_SCLAE=True,
    # Deep Iterative Matching
    CATRE=dict(
        NAME="CATRE",  # CATRE
        TASK="refine",  # refine | init | init+refine
        REFINE_SCLAE=True,  # 9D refine or 6D refine
        NUM_CLASSES=6,  # only valid for class aware
        N_ITER_TRAIN=4,
        N_ITER_TRAIN_WARM_EPOCH=4,  # linearly increase the refine iter from 1 to N_ITER_TRAIN until this epoch
        N_ITER_TEST=4,
        USE_MTL=False,  # uncertainty multi-task weighting
        ## backbone
        BACKBONE=dict(
            FREEZE=True,
        ),
        ## backbone (GCN3D)
        GCN3D=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="gcn3d",
                num_points=1500 + 32,
                neighbor_num=10,
                support_num=7,
            ),
        ),
        ## backbone (POINTNET)
        PCLNET=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="point_net",
                num_points=1500,
                global_feat=True,
                feature_transform=False,
                out_dim=1024,
            ),
        ),
        KPSNET=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="point_net",
                num_points=8,
                global_feat=True,
                out_dim=1024,
            ),
            LR_MULT=1.0,
        ),
        ## pose head for delta R/T/s
        POSE_HEAD=dict(
            FREEZE=False,
            ROT_TYPE="ego_rot6d",  # {ego|allo}_{quat|rot6d}
            CLASS_AWARE=False,
            INIT_CFG=dict(
                type="FC_RotTransHead",
                in_dim=1024 * 2,
                num_layers=2,
                feat_dim=256,
                norm="none",  # BN | GN | none
                num_gn_groups=32,
                act="gelu",  # relu | lrelu | silu (swish) | gelu | mish
            ),
            LR_MULT=1.0,
            DELTA_T_SPACE="image",  # image | 3D
            DELTA_T_WEIGHT=1.0,  # deepim-pytorch use 0.1 (for the version without K_zoom/zoom_factor)
            # deepim | cosypose (deepim's delta_z=0 means nochange, cosypose uses 1/exp, so 1 means nochamge)
            T_TRANSFORM_K_AWARE=True,  # whether to use zoomed K; deepim False | cosypose True
            DELTA_Z_STYLE="cosypose",
        ),
        ROT_HEAD=dict(
            FREEZE=False,
            ROT_TYPE="ego_rot6d",  # {ego|allo}_rot6d
            CLASS_AWARE=False,
            INIT_CFG=dict(
                type="DisentangledRotHead",
                in_dim=1286,
                rot_dim=3,  # ego_rot6d
                feat_flatten=False,
                norm_input=False,
                num_points=1,
                point_bias=True,
            ),
            LR_MULT=1.0,
            DELTA_T_SPACE="image",  # image | 3D
            DELTA_T_WEIGHT=1.0,  # deepim-pytorch use 0.1 (for the version without K_zoom/zoom_factor)
            # deepim | cosypose (deepim's delta_z=0 means nochange, cosypose uses 1/exp, so 1 means nochamge)
            T_TRANSFORM_K_AWARE=True,  # whether to use zoomed K; deepim False | cosypose True
            DELTA_Z_STYLE="cosypose",
            SCLAE_TYPE="iter_add",  # ${iter|mean}_${add|mul}
        ),
        NOCS_HEAD=dict(
            FREEZE=False,
            LR_MULT=1.0,
            INIT_CFG=dict(
                type="ConvPointNocsHead",
                in_dim=1286,
                feat_kernel_size=1,
                out_kernel_size=1,
                num_layers=2,
                feat_dim=256,
                norm="GN",  # BN | GN | none
                num_gn_groups=32,
                act="gelu",  # relu | lrelu | silu (swish) | gelu | mish
                last_act="sigmoid",
                norm_input=False,
                use_bias=False,
            ),
        ),
        T_HEAD=dict(
            WITH_KPS_FEATURE=True,
            FREEZE=False,
            INIT_CFG=dict(
                type="FC_TransHead",
                in_dim=1286,
            ),
            LR_MULT=1.0,
        ),
        S_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="FC_SizeHead",
                in_dim=1286,
            ),
            LR_MULT=1.0,
        ),
        TS_HEAD=dict(
            WITH_KPS_FEATURE=True,
            WITH_INIT_SCALE=False, # use scale as explicit input
            WITH_INIT_TRANS=False, # use scale as explicit input
            FREEZE=False,
            INIT_CFG=dict(
                type="ConvOutTransSizeHead",
                in_dim=1286,
            ),
            LR_MULT=1.0,
        ),
        LOSS_CFG=dict(
            # point matching loss ----------------
            PM_LOSS_TYPE="L1",  # L1 | Smooth_L1
            PM_SMOOTH_L1_BETA=1.0,
            PM_LOSS_SYM=False,  # use symmetric PM loss
            # if False, the trans loss is in point matching loss
            PM_R_ONLY=False,  # only do R loss in PM
            PM_WITH_SCALE=True,
            PM_DISENTANGLE_T=False,  # disentangle R/T
            PM_DISENTANGLE_Z=False,  # disentangle R/xy/z
            PM_T_USE_POINTS=True,  # only used for disentangled loss
            PM_USE_BBOX=False, # if true, the points is bbox
            PM_LW=1.0,
            # rot loss ----------------------------------
            ROT_LOSS_TYPE="angular",  # angular | L2
            ROT_YAXIS_LOSS_TYPE="L1",
            ROT_LW=0.0,
            ROT_LW_PP=0.0,
            # trans loss ----------------------------------
            TRANS_LOSS_TYPE="L1",
            TRANS_LOSS_DISENTANGLE=True,
            TRANS_LW=0.0,
            # scale loss ----------------------------------
            SCALE_LOSS_TYPE="L1",
            SCALE_LW=0.0,
            # nocs loss ----------------------------------
            NOCS_LOSS_TYPE="L1", # only support l1 for now
            NOCS_LW=0.0,
            SYM_NOCS_TYPE="YAXIS",  # YAXIS | CAPTRA | SYMROTATE
            NOCS_SYM_AWARE=True,
        ),
    ),
    # some d2 keys but not used
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False,
)

TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    SAVE_RESULTS_ONLY=False,
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),  # d2 keys, not used
)
