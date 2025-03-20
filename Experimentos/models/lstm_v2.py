import torch
import torch.nn as nn

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x):
        """
        x should be of shape (batch_size, seq_length, input_size), where input_size
        is the amount of features in each time step.

        The outputs of the LSTM layer are:
          - out: tensor of shape (batch_size, seq_length, hidden_size) containing
            the output features (h_t) from the last layer of the LSTM, for each t.
          - h_n: tensor of shape (n_layers, batch_size, hidden_size) containing
            the hidden state for each element in the sequence.
          - c_n: tensor of shape (n_layers, batch_size, hidden_size) containing
            the final cell state for each element in the sequence.
        """
        # Set initial hidden and cell states
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        _, (hidden_state, _) = self.lstm(x, (h0, c0))
        lstm_out = hidden_state[-1] # Use the last hidden_state
        return lstm_out


class LSTMClassifier_v2(nn.Module):
    def __init__(
        self, lstm_input_size, lstm_hidden_size, lstm_n_layers, n_static_feats, dropout
    ):
        super(LSTMClassifier_v2, self).__init__()
        self.lstm = LSTMBlock(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            n_layers=lstm_n_layers,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size+n_static_feats, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_temp, x_static):
        lstm_out = self.lstm(x_temp)
        x = torch.cat((lstm_out, x_static), dim=1)
        x = self.fc(x)
        return x


def define_lstm_v2_model(trial):
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.3, 0.8)
    return LSTMClassifier_v2(
        lstm_input_size=1,
        lstm_hidden_size=hidden_size,
        lstm_n_layers=n_layers,
        n_static_feats=1,
        dropout=dropout
    )
