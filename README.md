# MIPI2024_GoodGame
> This repository is the official [MIPI Challenge 2024](https://mipi-challenge.org/MIPI2024/) implementation of Team GoodGame in [Nighttime Flare Removal](https://codalab.lisn.upsaclay.fr/competitions/16998).
> The restoration results of the testing images can be downloaded from [here](https://pan.baidu.com/s/1amA5Xu_sPKJNEWwpFLkbiw?pwd=6666).
Our pretrained models can be downloaded from [here](https://pan.baidu.com/s/1POS3L6PsNWWC5787oTdiCg?pwd=6666).
## Usage
### Train
```
python train.py --data_source your_data_path --experiment you_experiment_name
```


### infer
```
python infer.py --data_source your_test_data_path --model ckpt_path 
```
