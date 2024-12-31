import torch
import torch.nn as nn

class LSTMClassifier_v2(nn.Module):
    """
    This neural network processes the dynamic variable through an LSTM and then
    adds the static variable to the end to make the classification.
    """
    
    def __init__(self, input_size, hidden_size, static_size, num_layers, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + static_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_dynamic, x_static):
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
        
        x_dynamic: (batch_size, seq_length, input_size)
        x_static: (batch_size, static_size)
        """
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x_dynamic.size(0), self.hidden_size).to(x_dynamic.device)
        c0 = torch.zeros(self.num_layers, x_dynamic.size(0), self.hidden_size).to(x_dynamic.device)

        # Forward propagate through LSTM
        _, (hidden_state, _) = self.lstm(x_dynamic, (h0, c0))
        # Use the last hidden_state
        lstm_out = hidden_state[-1]
        
        combined = torch.cat((lstm_out, x_static), dim=1)
        
        logits = self.fc(combined)
        return logits


def define_lstm_v2_model(trial):
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    num_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    return LSTMClassifier_v2(1, hidden_size, 1, num_layers, dropout)
