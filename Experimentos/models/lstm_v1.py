import torch
import torch.nn as nn

from constants import N_LAYERS, HIDDEN_SIZES, DROPOUTS
from models.blocks.lstm_block import LSTMBlock
from models.blocks.fc_block import FCBlock

class LSTMClassifier_v1(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, dropout):
        super().__init__()

        self.lstm = LSTMBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.fc = FCBlock(
            input_size=hidden_size,
            hidden_sizes=[128],
            dropout=dropout
        )

    def forward(self, x):
        lstm_out = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits


def define_lstm_v1_model(trial, input_size):
    hidden_size = trial.suggest_categorical("hidden_size", HIDDEN_SIZES)
    dropout = trial.suggest_categorical("dropout", DROPOUTS)
    return LSTMClassifier_v1(input_size, N_LAYERS, hidden_size, dropout)
