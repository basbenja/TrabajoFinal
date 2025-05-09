import torch.nn as nn

from constants import N_LAYERS

from models.blocks.fc_block import FCBlock

class DenseClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout):
        super(DenseClassifier, self).__init__()
        self.fc = FCBlock(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        )

    def forward(self, x):
        logits = self.fc(x)
        return logits


def define_dense_model(trial, input_size):
    hidden_sizes = [
        trial.suggest_categorical(f"n_units_l{i}", [16, 32, 64, 128]) for i in range(N_LAYERS)
    ]
    dropout = trial.suggest_categorical("dropout", [0.1, 0.3, 0.5, 0.7])
    return DenseClassifier(input_size, hidden_sizes, dropout)
