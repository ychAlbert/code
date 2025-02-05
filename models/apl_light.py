import torch
import torch.nn as nn
import torch.nn.functional as F

class LightSE(nn.Module):
    """Lightweight Squeeze-and-Excitation Module"""
    def __init__(self, channels):
        super(LightSE, self).__init__()
        reduced_channels = max(channels // 8, 8)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        # Depthwise
        nn.Conv1d(in_planes, in_planes, kernel_size=3, stride=stride, 
                 padding=1, groups=in_planes, bias=False),
        # Pointwise
        nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 使用深度可分离卷积替换标准卷积
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = LightSE(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # 添加SE注意力
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 使用1x1卷积降维
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        # 使用深度可分离卷积
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        # 使用1x1卷积升维
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.SiLU(inplace=True)
        self.se = LightSE(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # 添加SE注意力
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, inchannel=52, activity_num=6, location_num=16):
        super(ResNet, self).__init__()
        self.inplanes = 64  # 减少初始通道数
        
        # 轻量化stem层
        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannel, self.inplanes, kernel_size=3, stride=2, 
                     padding=1, bias=False),
            nn.BatchNorm1d(self.inplanes),
            nn.SiLU(inplace=True)
        )
        
        # 主干网络
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 轻量化活动分类器
        self.classifier_act = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, groups=256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, activity_num)
        )

        # 轻量化位置分类器
        self.classifier_loc = nn.Sequential(
            nn.Conv1d(512 * block.expansion, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, groups=256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, location_num)
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        # 特征提取
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类
        act_out = self.classifier_act(x)
        loc_out = self.classifier_loc(x)

        return act_out, loc_out