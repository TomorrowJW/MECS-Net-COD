import torch
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# 定义配置文件,写入一个配置类中
class Config():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = 1e-4  # 学习率
    weight_decay = 0
    num_epochs = 200
    clip = 0.5

    batch_size = 18
    train_size = 384
    test_size = 384
    num_workers = 10

    rgb_path = './data/TrainDataset/Imgs/'  # 训练rgb图片的路径(the path to image)
    GT_path = './data/TrainDataset/GT/'  # 训练标签的路径(the path to GT)
    Edge_path = './data/TrainDataset/Edge/'  #训练edge标签的路径(the path to Edge GT)

    test_path = './data/TestDataset' #测试集路径(the path to TestDataset)

    save_model_path = './save_models/res/'# 保存权重的路径(the path to save weights)
    save_results_path = './save_results/res/prediction/'# 保存预测图的路径(the path to save prediction maps)
    save_edge_results_path = './save_results/res/edges/'# 保存预测边缘图的路径(the path to save prediction edge maps)




