# Arquitectura basada en "LSTM Fully Convolutional Networks for Time Series
# Classification" (https://arxiv.org/abs/1709.05206)
import torch
import torch.nn as nn
import torch.nn.functional as F

# FCN: Fully Convolutional Network
class FCNBlock(nn.Module):
    def __init__(self):
        super(FCNBlock, self).__init__()
        # in_channels: number of input features per time step
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, padding=0)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=2, padding=0)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # Remove last dimension
        return x


class FCN_FC(nn.Module):
    def __init__(self, n_static_feats=1):
        super(FCN_FC, self).__init__()
        self.fcn = FCNBlock()
        # out_features=1 when using BCEWithLogitsLoss
        # out_features=2 when using CrossEntropyLoss
        self.fc = nn.Linear(in_features=128+n_static_feats, out_features=1)

    def forward(self, x_temp, x_static):
        x = self.fcn(x_temp)
        x = torch.cat((x, x_static), dim=1)
        x = self.fc(x)
        return x


def define_fcn_model(trial):
    ###### TO DO #######
    return FCN_FC(n_static_feats=1)