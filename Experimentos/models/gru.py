import torch
import torch.nn as nn

from constants import N_LAYERS

class GRUCLassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU Layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected layer to classify the output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through GRU
        out, _ = self.gru(x, h0)     # out: (batch_size, seq_length, hidden_size)

        # Use the last hidden state of the sequence for classification
        out = self.fc(out[:, -1, :])        # out: (batch_size, output_size)
        return out


def define_gru_model(trial, input_size):
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    dropout = trial.suggest_categorical("dropout", [0.3, 0.5, 0.7, 0.8])
    return GRUCLassifier(input_size, hidden_size, N_LAYERS, dropout)
