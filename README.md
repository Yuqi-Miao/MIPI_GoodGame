# MIPI2024_GoodGame
> This repository is the official [MIPI Challenge 2024](https://mipi-challenge.org/MIPI2024/) implementation of Team GoodGame in [https://mipi-challenge.org/MIPI2024/](https://codalab.lisn.upsaclay.fr/competitions/16998).
> The restoration results of the testing images can be downloaded from [here](https://pan.baidu.com/s/1p7SofDAIdL6VcpuEHOsiig?pwd=r8lp).
Our pretrained models can be downloaded from [here](https://pan.baidu.com/s/19JgYIaNSaF-b7mweqlb9rg?pwd=p5u8).
## Usage
### Train
```
./train.sh
```
Attention that you should change the path of datasets.
### Test
```
python main_test_swinir.py --task classical_sr --scale 4 --training_patch_size 48 --model_path your path --folder_lq your low-resolution image path --folder_gt your high-resolution image path
```
