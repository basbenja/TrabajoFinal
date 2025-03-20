import torch
import torch.nn as nn

class GRUCLassifier(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # GRU Layer
        self.gru = nn.GRU(
            input_size, hidden_size, n_layers, batch_first=True, dropout=dropout
        )

        # Fully connected layer to classify the output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through GRU
        out, _ = self.gru(x, h0)     # out: (batch_size, seq_length, hidden_size)

        # Use the last hidden state of the sequence for classification
        out = self.fc(out[:, -1, :])        # out: (batch_size, output_size)
        return out


def define_gru_model(trial, input_size):
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    n_layers = trial.suggest_int("n_layers", 1, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    return GRUCLassifier(input_size, hidden_size, n_layers, dropout)
