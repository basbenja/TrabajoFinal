import torch
import torch.nn as nn

from constants import N_LAYERS

class LSTMClassifier_v1(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, dropout):
        """
        input_size: amount of features of each time series
        num_layers: amount of LSTM layers
        hidden_size: amount of neurons per layer
        output_size: amount of classes to classify (2 if we are use CrossEntropyLoss, 
        1 if we use BCEWithLogitsLoss)
        dropout: dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Fully connected layer to classify the output
        self.fc = nn.Linear(hidden_size, out_features=1)

    def forward(self, x):
        """
        x should be of shape (batch_size, seq_length, input_size), where input_size
        is the amount of features in each time step.

        The outputs of the LSTM layer are:
          - out: tensor of shape (batch_size, seq_length, hidden_size) containing
            the output features (h_t) from the last layer of the LSTM, for each t.
          - h_n: tensor of shape (num_layers, batch_size, hidden_size) containing
            the hidden state for each element in the sequence.
          - c_n: tensor of shape (num_layers, batch_size, hidden_size) containing
            the final cell state for each element in the sequence.

        The fully connected layer is applied to the last hidden state of the sequence.
        """

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))     # out: (batch_size, seq_length, hidden_size)

        # out, _ = self.lstm(x)     # out: (batch_size, seq_length, hidden_size)

        # Use the last hidden state of the sequence for classification
        out = self.fc(out[:, -1, :])        # out: (batch_size, output_size)
        return out


def define_lstm_v1_model(trial, input_size):
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
    dropout = trial.suggest_categorical("dropout", [0.3, 0.5, 0.7, 0.8])
    return LSTMClassifier_v1(input_size, N_LAYERS, hidden_size, dropout)
