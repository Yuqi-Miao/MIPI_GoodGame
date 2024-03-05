# MIPI2024_GoodGame
> This repository is the official [MIPI Challenge 2024](https://mipi-challenge.org/MIPI2024/) implementation of Team GoodGame in [Nighttime Flare Removal](https://codalab.lisn.upsaclay.fr/competitions/16998).
> The restoration results of the testing images can be downloaded from [here](https://pan.baidu.com/s/1p7SofDAIdL6VcpuEHOsiig?pwd=r8lp).
Our pretrained models can be downloaded from [here](https://pan.baidu.com/s/19JgYIaNSaF-b7mweqlb9rg?pwd=p5u8).
## Usage
### Train
```
python train.py --data_source your_data_path --experiment you_experiment_name```
Attention that you should change the path of datasets.
### infer
```
python infer.py --data_source your_test_data_path --model ckpt_path 
```
