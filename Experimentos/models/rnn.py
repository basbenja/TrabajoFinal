import torch
import torch.nn as nn

class RNNCLassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN Layer
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        # Fully connected layer to classify the output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through GRU
        out, _ = self.rnn(x, h0)     # out: (batch_size, seq_length, hidden_size)

        # Use the last hidden state of the sequence for classification
        out = self.fc(out[:, -1, :])        # out: (batch_size, output_size)
        return out


def define_model(trial, input_size):
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    return RNNCLassifier(input_size, hidden_size, num_layers, dropout)

