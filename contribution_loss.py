import torch
import torch.nn as nn


class JointClassifyCrossLoss(nn.Module):
    """
    借鉴于完全平方公式（[a+b]^2=a^2+2ab+b^2）设计的损失函数
    """
    def __init__(self, weight, reduction='sum'):
        super(JointClassifyCrossLoss, self).__init__()
        self.loss_self = nn.CrossEntropyLoss(reduction=reduction)
        self.loss_cross = nn.BCELoss(reduction=reduction)
        self.weight = weight

    def forward(self, act_predict, act_target, loc_predict, loc_target):
        # 计算每个任务的交叉熵损失
        act_loss = self.loss_self(act_predict, act_target)
        loc_loss = self.loss_self(loc_predict, loc_target)
        # 计算每个任务的正确分类情况
        act_res = (act_predict.data.max(1)[1] == act_target).float()
        loc_res = (loc_predict.data.max(1)[1] == loc_target).float()
        # act_res = (act_predict == act_target).float()
        # loc_res = (loc_predict == loc_target).float()
        # 计算交叉损失
        joint_cross_loss = self.loss_cross(act_res, loc_res)
        # 计算总损失
        loss = torch.sqrt(act_loss.pow(2) + loc_loss.pow(2) + self.weight * joint_cross_loss)
        return loss


class JointClassifyFocalLoss(nn.Module):
    """
    借鉴于Focal Loss设计的损失函数
    """

    def __init__(self, alpha, gamma, reduction='sum'):
        super(JointClassifyFocalLoss, self).__init__()
        self.loss_self = nn.CrossEntropyLoss(reduction=reduction)
        self.loss_cross = nn.BCELoss(reduction=reduction)
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, act_predict, act_target, loc_predict, loc_target):
        # 计算每个任务的交叉熵损失
        act_loss = self.loss_self(act_predict, act_target)
        loc_loss = self.loss_self(loc_predict, loc_target)
        # 计算每个任务的正确分类情况
        act_res = (act_predict.data.max(1)[1] == act_target).float()
        loc_res = (loc_predict.data.max(1)[1] == loc_target).float()
        BCE_loss = self.loss_cross(act_res, loc_res)
        targets = loc_res.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** 2 * BCE_loss
        loss = act_loss + loc_loss + F_loss.mean()
        return loss
