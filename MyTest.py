import os
import torch
import torch.nn as nn
import numpy as np
from models.CTranNet_Swin_V2 import CL_Model
from utils.dataloader import test_dataset
from config import Config
import time
import imageio
import torch.nn.functional as F

cfg = Config()
model = CL_Model().to(cfg.device)
model = nn.DataParallel(model).to(cfg.device)
model.load_state_dict(torch.load('./save_models/' + 'swin_199.pth'))
model.cuda()
model.eval()

for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']: #'CAMO','CHAMELEON','COD10K'
    data_path = cfg.test_path + '{}/'.format(_data_name)
    save_path = cfg.save_results_path + '{}/'.format(_data_name)
    save_edge_path = cfg.save_edge_results_path + '{}/'.format(_data_name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_edge_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(cfg,image_root, gt_root, cfg.test_size)

    total_time = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        torch.cuda.synchronize()
        start = time.time()
        res,s2,s3,s4,s5,e,e2,e3,e4,e5 = model(image)
        torch.cuda.synchronize()
        end = time.time()
        total_time += end - start

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=True)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path + name, (res * 255).astype(np.uint8))
        e = F.interpolate(e, size=gt.shape, mode='bilinear', align_corners=True)
        e = e.sigmoid().data.cpu().numpy().squeeze()
        e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        imageio.imwrite(save_edge_path+name, (e*255).astype(np.uint8))

        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(total_time/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/total_time))