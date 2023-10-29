import torch
# torch.cuda.current_device()
# torch.cuda._initialized=True
import torch.nn as nn
import os
import random
import numpy as np
from utils.dataloader import get_loader
from utils.utils import clip_gradient, poly_lr, AvgMeter
import torch.nn.functional as F
from config import Config
#from models.CTranNet import CL_Model
from models.CTranNet_Swin_V2 import CL_Model #实现的是MECS-Net的Swin版本(implemented is the Swin version of MECS-Net)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(42)

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_loss(predict_, target):
    predict = torch.sigmoid(predict_)
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def cross_entropy2d_edge(input, target, reduction='mean'):
    assert (input.size()) == target.size()
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha *pos + beta*neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

cfg = Config() #实例化配置类

model = CL_Model().to(cfg.device)
#model = nn.DataParallel(model).to(cfg.device) #我们使用的是两张显卡V100，每张卡只有16G

#统计模型的参数量
total1 = sum([param.nelement() for param in model.parameters()])
print('Number of parameter : %.3fM'%(total1/1e6))

train_dataloader = get_loader(cfg, cfg.rgb_path, cfg.GT_path, cfg.Edge_path, cfg.batch_size, cfg.train_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
total_step = len(train_dataloader)
params = model.parameters()
optimizer = torch.optim.AdamW(params, cfg.lr, betas=(0.9, 0.999), eps=1e-08)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 学习率调节器，这里采用阶段调节

def train():
    print('Let us start to train the model:')

    for epoch in range(cfg.num_epochs):
        model.train()

        for i, data in enumerate(train_dataloader, start=1):  # 下标从1开始
            optimizer.zero_grad()  # 每一步都要将之前的梯度进行清零

            images, gts, edges = data
            images = images.to(cfg.device)
            gts = gts.to(cfg.device)
            edges = edges.to(cfg.device)

            s1,s2,s3,s4,s5, e1,e2,e3,e4,e5 = model(images)


            loss_ce = 1*(structure_loss(s1,gts)+structure_loss(s2,gts)\
                           + structure_loss(s3, gts)+structure_loss(s4,gts)+structure_loss(s5,gts))

            loss_ed =  1 * ( dice_loss(e1, edges) + cross_entropy2d_edge(e1, edges)+
                             dice_loss(e2, edges) + cross_entropy2d_edge(e2, edges) \
                    + dice_loss(e3, edges) + cross_entropy2d_edge(e3, edges) \
                    + dice_loss(e4, edges) + cross_entropy2d_edge(e4, edges) \
                    + dice_loss(e5, edges) + cross_entropy2d_edge(e5, edges))

            loss = loss_ce + loss_ed

            loss.backward()

            clip_gradient(optimizer, cfg.clip)
            optimizer.step()

            if i % 200 == 0 or i == total_step:
                print(
                    'Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Lossce: {:.8f}, Lossed: {:.8f}, Loss:{:.8f}'.
                    format(epoch, cfg.num_epochs, i, total_step, optimizer.param_groups[0]['lr'], loss_ce.item(),
                           loss_ed.item(), loss.item()))

        scheduler.step()
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), cfg.save_model_path + '%d' % epoch + 'CL.pth')


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    train()
