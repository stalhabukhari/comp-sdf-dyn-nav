# Differentiable Composite Neural Signed Distance Fields for Robot Navigation in Dynamic Indoor Environments

<!-- Shields-->
[<img src="https://img.shields.io/badge/Website-%230077B5.svg?&style=plastic&logo=home-assistant&logoColor=white&labelColor=black&color=white" />](https://stalhabukhari.github.io/icra25-sdf-dyn-nav)
[<img src="https://img.shields.io/badge/Paper-%230077B5.svg?&style=plastic&logo=arxiv&labelColor=ff0000&color=ffffff" />](https://arxiv.org/abs/2502.02664)
[<img src="https://img.shields.io/badge/Data-%230077B5.svg?&style=plastic&logo=google-drive&labelColor=white&color=blue" />](https://drive.google.com/drive/folders/1RxTsU6Mlks7N4nMjbm9Yxg_DcJIhpscT?usp=sharing)


## Setup

```shell
git clone --recursive https://github.com/stalhabukhari/obj-comp-sdf-dyn-nav.git
conda create -n sdf-nav python=3.9
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
# yolov5
pip install -r yolov5/requirements.txt
# CUDA-based farthest point sampling (optional)
pip install Pointnet2_PyTorch/pointnet2_ops_lib/
```

Congfigure the iGibson environment: https://github.com/stalhabukhari/iGibson.git


## Usage

Train SDF models using the following repositories:

- Object-level SDF: DeepSDF: https://github.com/stalhabukhari/DeepSDF
- Scene-level SDF: https://github.com/stalhabukhari/iSDF

Pretrained models are available on the Google Drive link at the top.

Simulations can be executed via: `python sim_<method>.py --cfg <path-to-config>`

Examples:

```shell
CFG=configs/robs-cfgs/robs.yaml
# dual mode
python sim_dual_mode.py --cfg $CFG --dynamic
# robot sdf
python sim_robot_sdf.py --cfg $CFG --dynamic
# scene sdf
python sim_scene_sdf.py --cfg $CFG --dynamic
```


## Citation

```bibtex
@inproceedings{bukhari25icra,
  title={Differentiable Composite Neural Signed Distance Fields for Robot Navigation in Dynamic Indoor Environments},
  author={Bukhari, S. Talha and Lawson, Daniel and Qureshi, Ahmed H.},
  booktitle={2025 International Conference on Robotics and Automation (ICRA)},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgement

We thank the authors of the following repositories, which we adapt code from:

- https://github.com/facebookresearch/iSDF
- https://github.com/facebookresearch/DeepSDF
- https://github.com/ultralytics/yolov5
- https://github.com/erikwijmans/Pointnet2_PyTorch


## License

Code is released under the MIT License. See the LICENSE file for more details.
