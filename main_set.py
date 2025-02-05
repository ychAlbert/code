import random
import argparse

import numpy as np
import torch

# ----------------------------------------------------------------------------------------------------------------------
# 实验随机性设置
# ----------------------------------------------------------------------------------------------------------------------
# 随机数种子
_seed_ = 202411
# 设置Python 的随机数生成器的种子。这将确保随机数生成器生成的随机序列是可预测的
random.seed(_seed_)
# 设置NumPy的随机数生成器的种子。这将确保在使用NumPy进行随机操作时得到可重复的结果
np.random.seed(_seed_)
# 设置PyTorch的随机数生成器的种子。这将确保在使用PyTorch进行随机操作时得到可重复的结果
torch.random.manual_seed(_seed_)
# 设置所有可用的CUDA设备的随机数生成器的种子。这将确保在使用CUDA加速时得到可重复的结果
torch.cuda.manual_seed_all(_seed_)
# 将CuDNN的随机性设置为确定性模式。这将确保在使用CuDNN加速时得到可
torch.backends.cudnn.deterministic = True
# 禁用CuDNN的自动寻找最佳卷积算法。这将确保在使用CuDNN加速时得到可重复的结果。
torch.backends.cudnn.benchmark = False
# 设置PyTorch进行CPU多线程并行计算时所占用的线程数，用来限制PyTorch所占用的CPU数目
torch.set_num_threads(4)

# ----------------------------------------------------------------------------------------------------------------------
# 实验参数设置
# ----------------------------------------------------------------------------------------------------------------------
# 获取命令行参数解析对象
parser = argparse.ArgumentParser()
parser.add_argument('--loss_type', type=str, default='cross', choices=['origin', 'cross', 'focal'], help='使用的loss类型')
parser.add_argument('--cross_weight', type=float, default=2, help='自定义联合loss的weight参数')
parser.add_argument('--focal_alpha', type=float, default=0.5, help='自定义联合loss的alpha参数')
parser.add_argument('--focal_gamma', type=float, default=2, help='自定义联合loss的gamma参数')
parser.add_argument('--data_format', type=str, default='joint', help='是否使用修改的数据格式')
parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else 'cpu'))
# 解析命令行参数
args = parser.parse_args()
