import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    """Squeeze-and-Excitation Module"""
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_att = SEModule(channels)
        self.spatial_att = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.channel_att(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.spatial_att(y)
        return x * y

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, use_se=True, use_cbam=False):
        super(BasicBlock, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv3x3(inplanes, width, stride, groups, dilation)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = conv3x3(width, width, groups=groups, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(width)
        self.relu = nn.SiLU(inplace=True)  # Replace ReLU with SiLU
        self.downsample = downsample
        self.stride = stride
        
        # Attention modules
        self.se = SEModule(width) if use_se else None
        self.cbam = CBAM(width) if use_cbam else None
        
        # Stochastic Depth
        self.drop_path = nn.Dropout(p=0.1) if stride == 1 else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)
        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.drop_path is not None:
            out = self.drop_path(out)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, use_se=True, use_cbam=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # Attention modules
        self.se = SEModule(planes * self.expansion) if use_se else None
        self.cbam = CBAM(planes * self.expansion) if use_cbam else None
        
        # Stochastic Depth
        self.drop_path = nn.Dropout(p=0.1) if stride == 1 else None

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

        if self.se is not None:
            out = self.se(out)
        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.drop_path is not None:
            out = self.drop_path(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, inchannel=52, activity_num=6, location_num=16, 
                 groups=1, width_per_group=64, use_se=True, use_cbam=False):
        super(ResNet, self).__init__()
        
        self.inplanes = 128
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        # Stem network
        self.conv1 = nn.Sequential(
            nn.Conv1d(inchannel, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True)
        )
        
        # Main layers
        self.layer1 = self._make_layer(block, 128, layers[0], use_se=use_se, use_cbam=use_cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_se=use_se, use_cbam=use_cbam)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_se=use_se, use_cbam=use_cbam)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_se=use_se, use_cbam=use_cbam)

        # Activity classifier with feature pyramid
        self.fpn_act = FeaturePyramidNetwork(
            [128 * block.expansion, 128 * block.expansion,
             256 * block.expansion, 512 * block.expansion],
            256
        )
        self.classifier_act = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, activity_num)
        )

        # Location classifier with feature pyramid
        self.fpn_loc = FeaturePyramidNetwork(
            [128 * block.expansion, 128 * block.expansion,
             256 * block.expansion, 512 * block.expansion],
            256
        )
        self.classifier_loc = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, location_num)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, use_cbam=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                          self.base_width, self.dilation, use_se, use_cbam))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, dilation=self.dilation,
                              use_se=use_se, use_cbam=use_cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        # Get features from each stage
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Activity classification path
        act_features = self.fpn_act([c1, c2, c3, c4])
        act_out = self.classifier_act(act_features[-1])

        # Location classification path
        loc_features = self.fpn_loc([c1, c2, c3, c4])
        loc_out = self.classifier_loc(loc_features[-1])

        return act_out, loc_out

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block_module = nn.Conv1d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv1d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

    def forward(self, x):
        features = []
        last_inner = self.inner_blocks[-1](x[-1])
        features.append(self.layer_blocks[-1](last_inner))

        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            inner_top_down = F.interpolate(last_inner, size=feature.shape[-1],
                                         mode='linear', align_corners=False)
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            features.append(layer_block(last_inner))

        return features[::-1]