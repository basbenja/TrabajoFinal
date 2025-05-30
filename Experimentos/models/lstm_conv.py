import torch
import torch.nn as nn

from constants import N_LAYERS, HIDDEN_SIZES, DROPOUTS

from models.blocks.lstm_block import LSTMBlock
from models.blocks.conv_block import ConvBlock
from models.blocks.fc_block import FCBlock

class LSTMConvClassifier(nn.Module):
    def __init__(
        self, lstm_input_size, lstm_hidden_size, lstm_num_layers, n_static_feats, dropout, conv_out_dim
    ):
        super(LSTMConvClassifier, self).__init__()

        self.lstm = LSTMBlock(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout
        )
        self.conv = ConvBlock(dropout=dropout)
        self.fc = FCBlock(
            input_size=conv_out_dim+lstm_hidden_size+n_static_feats,
            hidden_sizes=[128],
            dropout=dropout
        )
    
    def forward(self, x_temp, x_static):
        x_temp_for_lstm = x_temp.permute(0, 2, 1)
        lstm_out = self.lstm(x_temp_for_lstm)
        conv_out = self.conv(x_temp)
        combined = torch.cat((lstm_out, conv_out, x_static), dim=1)
        logits = self.fc(combined)
        return logits


def define_lstm_conv_model(trial, input_size, conv_out_dim):
    hidden_size = trial.suggest_categorical("hidden_size", HIDDEN_SIZES)
    dropout = trial.suggest_categorical("dropout", DROPOUTS)
    return LSTMConvClassifier(
        lstm_input_size=input_size,
        lstm_hidden_size=hidden_size,
        lstm_num_layers=N_LAYERS,
        n_static_feats=1,
        dropout=dropout,
        conv_out_dim=conv_out_dim
    )