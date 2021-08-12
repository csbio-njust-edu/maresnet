
"""
    saresnet in pytorch
    [1] Long-Chen Shen, Yan Liu, Jiangning Song, Dong-Jun Yu
    self-attention residual network for predicting DNA-protein binding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActResidualUnit(nn.Module):
    """PreAct Residual Unit
    Args:
        in_channels: residual unit input channel number
        out_channels: residual unit output channel numebr
        stride: stride of residual unit when stride = 2, down sample the feature map
    """

    def __init__(self, in_channels, out_channels, stride, is_turn=False):
        super().__init__()
        inner_channels = in_channels
        self.residual_function = nn.Sequential(
            # 1x3 conv
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, inner_channels, 3, stride=stride, padding=1, bias=False),

            # 1x3 conv
            nn.BatchNorm1d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inner_channels, out_channels, 3, stride=1, padding=1, bias=False)
        )

        self.shortcut = nn.Sequential()
        if stride != 2 or (in_channels != out_channels) or is_turn:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        res = self.residual_function(x)
        shortcut = self.shortcut(x)
        return res + shortcut


class AttentionModule1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, p=1, t=2, r=1):
        super().__init__()
        # """The hyper parameter p denotes the number of preprocessing Residual
        # Units before splitting into trunk branch and mask branch. t denotes
        # the number of Residual Units in trunk branch. r denotes the number of
        # Residual Units between adjacent pooling layer in the mask branch."""
        assert in_channels == out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown3 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup4 = self._make_residual(in_channels, out_channels, r)

        self.shortcut_short = PreActResidualUnit(in_channels, out_channels, 1)
        self.shortcut_long = PreActResidualUnit(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)  # [batch_size, 32, 51]
        input_size = (x.size(2))
        x_t = self.trunk(x)  # [batch_size, 32, 51]

        # first downsample out 26
        x_s = F.max_pool1d(x, kernel_size=self.kernel_size, stride=2, padding=self.padding)  # [batch_size, 256, 26]
        x_s = self.soft_resdown1(x_s)  # [batch_size, 32, 26]

        # 26 shortcut
        shape1 = (x_s.size(2))
        shortcut_long = self.shortcut_long(x_s)  # [batch_size, 32, 26]

        # seccond downsample out 13
        x_s = F.max_pool1d(x_s, kernel_size=self.kernel_size, stride=2, padding=self.padding)  # [batch_size, 32, 13]
        x_s = self.soft_resdown2(x_s)  # [batch_size, 32, 13]

        # 13 shortcut
        shape2 = (x_s.size(2))
        shortcut_short = self.soft_resdown3(x_s)

        # third downsample out 7
        x_s = F.max_pool1d(x_s, kernel_size=self.kernel_size, stride=2, padding=self.padding)  # [batch_size, 32, 7]
        x_s = self.soft_resdown3(x_s)  # [batch_size, 32, 7]

        # first upsample out 13
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape2)
        x_s += shortcut_short

        # second upsample out 26
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut_long

        # thrid upsample out 51
        x_s = self.soft_resup4(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)  # [batch_size, 32, 51]

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))

        return nn.Sequential(*layers)


class AttentionModule2(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()
        # """The hyperparameter p denotes the number of preprocessing Residual
        # Units before splitting into trunk branch and mask branch. t denotes
        # the number of Residual Units in trunk branch. r denotes the number of
        # Residual Units between adjacent pooling layer in the mask branch."""
        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resdown2 = self._make_residual(in_channels, out_channels, r)

        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup3 = self._make_residual(in_channels, out_channels, r)

        self.shortcut = PreActResidualUnit(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)  # [batch_size, 64, 26]
        input_size = (x.size(2))

        x_t = self.trunk(x)

        # first downsample out 13
        x_s = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        # 13 shortcut
        shape1 = (x_s.size(2))
        shortcut = self.shortcut(x_s)

        # seccond downsample out 7
        x_s = F.max_pool1d(x_s, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown2(x_s)

        # first upsample out 13
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=shape1)
        x_s += shortcut

        # second upsample out 26
        x_s = self.soft_resup3(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))
        return nn.Sequential(*layers)


class AttentionModule3(nn.Module):

    def __init__(self, in_channels, out_channels, p=1, t=2, r=1):
        super().__init__()

        assert in_channels == out_channels

        self.pre = self._make_residual(in_channels, out_channels, p)
        self.trunk = self._make_residual(in_channels, out_channels, t)
        self.soft_resdown1 = self._make_residual(in_channels, out_channels, r)
        self.soft_resup2 = self._make_residual(in_channels, out_channels, r)

        self.shortcut = PreActResidualUnit(in_channels, out_channels, 1)

        self.sigmoid = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.last = self._make_residual(in_channels, out_channels, p)

    def forward(self, x):
        x = self.pre(x)
        input_size = (x.size(2))

        x_t = self.trunk(x)

        # first downsample out 13
        x_s = F.max_pool1d(x, kernel_size=3, stride=2, padding=1)
        x_s = self.soft_resdown1(x_s)

        # first upsample out 13
        x_s = self.soft_resup2(x_s)
        x_s = F.interpolate(x_s, size=input_size)

        x_s = self.sigmoid(x_s)
        x = (1 + x_s) * x_t
        x = self.last(x)

        return x

    def _make_residual(self, in_channels, out_channels, p):

        layers = []
        for _ in range(p):
            layers.append(PreActResidualUnit(in_channels, out_channels, 1))

        return nn.Sequential(*layers)


class Attention(nn.Module):
    """residual attention netowrk
    Args:
        block_num: attention module number for each stage
    """
    def __init__(self, block_num, class_num=2):

        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1_1 = self._make_stage_1(32, 32, 3, block_num[0], AttentionModule1)
        self.stage1_2 = self._make_stage_1(32, 32, 5, block_num[0], AttentionModule1)
        self.stage1_3 = self._make_stage_1(32, 32, 9, block_num[0], AttentionModule1)
        self.stage2 = self._make_stage(96, 64, block_num[1], AttentionModule2)
        self.stage3 = self._make_stage(64, 64, block_num[2], AttentionModule3)
        self.stage4 = nn.Sequential(
            PreActResidualUnit(64, 64, 2, is_turn=True),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(64, 2)
        self.Dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pre_conv(x)  # [batch_size, 32, 101]
        x_1 = self.stage1_1(x)  # [batch_size, 32, 51]
        x_2 = self.stage1_2(x)  # [batch_size, 32, 51]
        x_3 = self.stage1_3(x)  # [batch_size, 32, 51]
        x = torch.cat((x_1, x_2, x_3), 1)

        x = self.stage2(x)  # [batch_size, 64, 26]
        x = self.stage3(x)  # [batch_size, 64, 13]
        x = self.stage4(x)  # [batch_size, 64, 7]
        x = self.avg(x)
        x = self.Dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def _make_stage(self, in_channels, out_channels, num, block):

        layers = []
        layers.append(PreActResidualUnit(in_channels, out_channels, 2, is_turn=True))

        for _ in range(num):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_stage_1(self, in_channels, out_channels, kernel_size, num, block):

        layers = []
        layers.append(PreActResidualUnit(in_channels, out_channels, 2, is_turn=True))
        for _ in range(num):
            layers.append(block(out_channels, out_channels, kernel_size, padding=int(kernel_size/2)))
        return nn.Sequential(*layers)


def multi_attention_resnet():
    return Attention([1, 1, 1])


