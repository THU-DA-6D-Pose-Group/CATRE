# CATRE

This repo provides for the implementation of the ECCV'22 paper:

**CATRE: Iterative Point Clouds Alignment for Category-level Object Pose Refinement**[bibtex(#Citation)]

## Dependencies

See [INSTALL.md](./docs/INSTALL.md)

## Datasets
```bash
datasets/
├── NOCS
    ├──REAL
        ├── real_test  # download from http://download.cs.stanford.edu/orion/nocs/real_test.zip
        ├── real_train # download from  http://download.cs.stanford.edu/orion/nocs/real_train.zip
        └── image_set  # we provide
    ├──gts             # download from http://download.cs.stanford.edu/orion/nocs/gts.zip
        └── real_test
    ├──test_init_poses # we provide 
    └──object_models   # we provide some necesarry files, complete files can be download from http://download.cs.stanford.edu/orion/nocs/obj_models.zip
```

Run python scripts to prepare the datasets. (Modified from https://github.com/mentian/object-deformnet)
```bash
# NOTE: this code will directly modify the data
cd $ROOT/preprocess
python shape_data.py
python pose_data.py
```

## Reproduce the results in the paper

```
./core/catre/test_catre.sh configs/catre/NOCS_REAL/aug05_kpsMS_r9d_catreDisR_shared_tspcl_convPerRot_scaleexp_120e.py 1  output/catre/NOCS_REAL/aug05_kpsMS_r9d_catreDisR_shared_tspcl_convPerRot_scaleexp_120e/model_final_wo_optim-82cf930e.pth
```

**NOTE:**

We fix a bug in the evaluation code of IOU, see \todo{link} for details.

## Training

`./core/catre/train_catre.sh configs/catre/NOCS_REAL/aug05_kpsMS_r9d_catreDisR_shared_tspcl_convPerRot_scaleexp_120e.py <gpu_ids> (other args)`

## Testing
`./core/catre/test_catre.sh configs/catre/NOCS_REAL/aug05_kpsMS_r9d_catreDisR_shared_tspcl_convPerRot_scaleexp_120e.py <gpu_ids> <ckpt_path> (other args)`

## Citation
If you find this repo useful in your research, please consider citing:
```
@InProceedings{liu_2022_catre,
  title     = {CATRE: Iterative Point Clouds Alignment for Category-level Object Pose Refinement},
  author    = {Liu, Xingyu and Wang, Gu and Li, Yi and Ji, Xiangyang},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  month     = {August},
  year      = {2022}
}
```