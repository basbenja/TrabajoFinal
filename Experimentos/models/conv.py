import torch
import torch.nn as nn

from constants import DROPOUTS

from models.blocks.conv_block import ConvBlock
from models.blocks.fc_block import FCBlock

class Conv_FC(nn.Module):
    def __init__(self, dropout, n_static_feats, conv_out_dim):
        super(Conv_FC, self).__init__()

        self.conv = ConvBlock(dropout=dropout)
        self.fc = FCBlock(
            input_size=conv_out_dim+n_static_feats,
            hidden_sizes=[128],
            dropout=dropout
        )


    def forward(self, x_temp, x_static):
        conv_out = self.conv(x_temp)
        combined = torch.cat((conv_out, x_static), dim=1)
        logits = self.fc(combined)
        return logits


def define_conv_model(trial, n_static_feats, conv_out_dim):
    dropout = trial.suggest_categorical("dropout", DROPOUTS)
    return Conv_FC(dropout, n_static_feats, conv_out_dim)