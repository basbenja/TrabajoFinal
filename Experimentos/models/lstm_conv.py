import torch
import torch.nn as nn

from constants import N_LAYERS

from blocks.lstm_block import LSTMBlock
from blocks.conv_block import ConvBlock
from blocks.fc_block import FCBlock

class LSTMConvClassifier(nn.Module):
    def __init__(
        self, lstm_input_size, lstm_hidden_size, lstm_num_layers, n_static_feats, dropout
    ):
        super(LSTMConvClassifier, self).__init__()

        self.lstm = LSTMBlock(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout
        )
        self.conv = ConvBlock()
        self.fc = FCBlock(
            input_size=lstm_hidden_size+n_static_feats,
            hidden_sizes=[128],
            dropout=dropout
        )
    
    def forward(self, x_temp, x_static):
        ## TODO
        pass


def define_lstm_conv_model(trial, input_size):
    ## TODO
    pass