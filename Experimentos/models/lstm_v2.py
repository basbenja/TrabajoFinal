import torch
import torch.nn as nn

from constants import N_LAYERS

from models.blocks.lstm_block import LSTMBlock
from models.blocks.fc_block import FCBlock

class LSTMClassifier_v2(nn.Module):
    def __init__(
        self, lstm_input_size, lstm_hidden_size, lstm_num_layers, n_static_feats, dropout
    ):
        super(LSTMClassifier_v2, self).__init__()

        self.lstm = LSTMBlock(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout
        )
        self.fc = FCBlock(
            input_size=lstm_hidden_size+n_static_feats,
            hidden_sizes=[128],
            dropout=dropout
        )

    def forward(self, x_temp, x_static):
        lstm_out = self.lstm(x_temp)
        combined = torch.cat((lstm_out, x_static), dim=1)
        logits = self.fc(combined)
        return logits


def define_lstm_v2_model(trial, input_size):
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    dropout = trial.suggest_categorical("dropout", [0.3, 0.5, 0.7, 0.8])
    return LSTMClassifier_v2(
        lstm_input_size=input_size,
        lstm_hidden_size=hidden_size,
        lstm_num_layers=N_LAYERS,
        n_static_feats=1,
        dropout=dropout
    )
