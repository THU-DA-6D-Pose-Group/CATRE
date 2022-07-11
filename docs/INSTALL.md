# Installation

* CUDA >= 10.1, Ubuntu >= 16.04

* Python >= 3.6, PyTorch >= 1.9, torchvision
    ```
    ## create a new environment
    conda create -n py37 python=3.7.4
    conda activate py37  # maybe add this line to the end of ~/.bashrc
    conda install ipython
    ## install pytorch: https://pytorch.org/get-started/locally/  (check cuda version)
    pip install torchvision -U  # will also install corresponding torch
    ```

* `detectron2` from [source](https://github.com/facebookresearch/detectron2).
    ```
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2
    pip install ninja
    pip install -e .
    ```

* `sh scripts/install_deps.sh`
