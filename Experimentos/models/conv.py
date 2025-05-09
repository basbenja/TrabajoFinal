# Arquitectura basada en "LSTM Fully Convolutional Networks for Time Series
# Classification" (https://arxiv.org/abs/1709.05206)
import torch
import torch.nn as nn

from models.blocks.conv_block import ConvBlock
from models.blocks.fc_block import FCBlock

class Conv_FC(nn.Module):
    def __init__(self, n_static_feats=1):
        super(Conv_FC, self).__init__()

        self.conv = ConvBlock()
        self.fc = FCBlock(
            input_size=128+n_static_feats,
            hidden_sizes=[128],
            dropout=0
        )


    def forward(self, x_temp, x_static):
        conv_out = self.conv(x_temp)
        combined = torch.cat((conv_out, x_static), dim=1)
        logits = self.fc(combined)
        return logits


def define_conv_model(trial, n_static_feats):
    ###### TO DO #######
    return Conv_FC(n_static_feats)