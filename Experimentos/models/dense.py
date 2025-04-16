import torch.nn as nn

from constants import N_LAYERS

class DenseClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout):
        super(DenseClassifier, self).__init__()
        backbone = []
        for hidden_size in hidden_sizes:
            backbone.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_size = hidden_size
        backbone.append(nn.Linear(hidden_sizes[-1], 1))
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        logits = self.backbone(x)
        return logits


def define_dense_model(trial, input_size):
    hidden_sizes = [
        trial.suggest_categorical(f"n_units_l{i}", [16, 32, 64, 128]) for i in range(N_LAYERS)
    ]
    dropout = trial.suggest_categorical("dropout", [0.1, 0.3, 0.5, 0.7])
    return DenseClassifier(input_size, hidden_sizes, dropout)
