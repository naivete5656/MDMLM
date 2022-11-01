# Spatial-Temporal Mitosis Detection in Phase-Contrast Microscopy via Likelihood Map Estimation by 3DCNN

[[Home]](http://human.ait.kyushu-u.ac.jp/index-e.html) [[Paper]](https://arxiv.org/abs/2004.12531) 

![Illustration](mitosisdetection_overview.png)

## Prerequisites
* System (tested on Ubuntu 18.04LTS)
* NVIDIA driver 430
* [Python>=3.6](https://www.python.org)
* [PyTorch>=0.4](https://pytorch.org)
* [MATLAB](https://jp.mathworks.com/products/matlab.html)

## Installation
Python setting

### Conda user
```bash
conda env create -f=requirement.yml
conda activate pytorch
```

### Docker user
```besh
docker build ./docker
sh run_docker.sh
```

## Data Preparation 

CVPR 2019 Contest on Mitosis Detection in Phase Contrast Microscopy Image Sequences

To use dataset, prease follow the guideline.
Now the dataset line was expired. If you want to use the dataset, please ask the contest organizers directly.
https://ieeexplore.ieee.org/abstract/document/9328484?casa_token=XLj19UfXiEwAAAAA:TdwkxaQwKywNwzsnDje3GgSL6960XqGUxNVLLXu2RBpWyb85DTy2f1TEqJJYYa4E9SmVjrfEzg

## How to use
1. Candidate path image extraction based on the brightness

  ```matlab
  matlab -nodesktop -nosplash -r "candidate_extractor(dataset_directory, './output/')"
  ```

1. Generate ground truth from candidate

  ```python
  python generate_ground_truth.py
  ```

1. Train V-Net

  ```python
  python train.py
  ```

1. Prediction by V-Net
  ```python
  python predict.py
  ```

## Citation
If you use this code for your research, please cite:
```bibtex
@article{nishimura2020spatial,
  title={Spatial-Temporal Mitosis Detection in Phase-Contrast Microscopy via Likelihood Map Estimation by 3DCNN},
  author={Nishimura, Kazuya and Bise, Ryoma},
  journal={arXiv preprint arXiv:2004.12531},
  year={2020}
}
```
