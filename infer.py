import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import *
from options import TestOptions
from model import Restormer
from dataset import SingleDataset
from collections import OrderedDict

print('---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------')
opt = TestOptions().parse()

image_dir = opt.outputs_dir + '/' + opt.experiment + '/infer'
# 当save_image为True时，删除文件夹下的所有文件 但是正常情况下都是原文件覆盖了
clean_dir(image_dir, delete=opt.save_image)

print('---------------------------------------- step 2/4 : data loading... ------------------------------------------------')
print('inferring data loading...')
infer_dataset = SingleDataset(datasource=opt.data_source)
infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)
print('successfully loading inferring pairs. =====> qty:{}'.format(len(infer_dataset)))

print('---------------------------------------- step 3/4 : model defining... ----------------------------------------------')
# 定义模型并将其移动到GPU上
model = Restormer().cuda()

# 使用 DataParallel 包装模型以在两个GPU上运行
model = torch.nn.DataParallel(model, device_ids=[0, 1])

print_para_num(model)
# 加载保存的状态字典
checkpoint = torch.load(opt.model_path)
state_dict = checkpoint['model_state_dict']

if not list(state_dict.keys())[0].startswith('module.'):
    # 使用字典推导式为每个键添加 'module.' 前缀
    state_dict = {'module.' + k: v for k, v in state_dict.items()}

# 加载调整后的状态字典到模型中
model.load_state_dict(state_dict)

print('successfully loading pretrained model.')

print('---------------------------------------- step 4/4 : testing... ----------------------------------------------------')

def main():
    model.eval()

    time_meter = AverageMeter()

    for i, (img, path) in enumerate(infer_dataloader):
        img = img.cuda()
        h, w = img.size(2), img.size(3)
        img = check_padding(img)

        with torch.no_grad():
            start_time = time.time()
            pred = model(img)
            times = time.time() - start_time

        pred_clip = torch.clamp(pred, 0, 1)
        pred_clip = pred[:, :, :h, :w]

        time_meter.update(times, 1)

        print('Iteration: ' + str(i+1) + '/' + str(len(infer_dataset)) + '  Processing image... ' + str(path) + '  Time ' + str(times))
        
        save_image(pred_clip, image_dir + '/' + os.path.basename(path[0]))
        # 清空未使用的缓存内存
        torch.cuda.empty_cache()

    print('Avg time: ' + str(time_meter.average()))

if __name__ == '__main__':
    main()