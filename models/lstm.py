import torch
import torch.nn as nn
import optuna

from torch.utils.data import DataLoader
from utils.train_predict import train_step, validate_step, predict
from utils.metrics import *

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # TODO: agregar Dropout

        # Fully connected layer to classify the output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))     # out: (batch_size, seq_length, hidden_size)

        # Use the last hidden state of the sequence for classification
        out = self.fc(out[:, -1, :])        # out: (batch_size, output_size)
        return out


def define_model(trial, input_size=2, output_size=1):
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    num_layers = trial.suggest_int("n_layers", 1, 3)
    return LSTMClassifier(input_size, hidden_size, output_size, num_layers)

def objective(trial, train_set, valid_set, class_weights, metrics, **kwargs):
    check_metrics(metrics, **kwargs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = define_model(trial).to(device)
    
    # Optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    
    # Epochs
    epochs = trial.suggest_int("n_epochs", 100, 300)
    
    # Batch Size
    batch_size = trial.suggest_int("batch_size", 16, 128)
    
    # Create the loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    
    # Get the averages for each period for the train loader
    X_train_tensor, y_train_tensor = train_set.tensors
    if 'avg_feats_diff' in metrics:
        train_features_mean = get_features_mean(X_train_tensor, y_train_tensor).to(device)

    # Get the X_valid_tensor so then we can calculate the features mean for the
    # individuals of the validation set that the model predicted as 1
    X_valid_tensor, _ = valid_set.tensors
    X_valid_tensor = X_valid_tensor.to(device)

    # Get the y_valid_tensor tensors so then we can calculate metrics
    _, y_valid_tensor = valid_set.tensors

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(class_weights[1], dtype=torch.float32)
    )

    for _ in range(epochs):
        train_step(model, train_loader, loss_fn, optimizer)
        
        logits = model(X_valid_tensor)
        y_valid_pred = predict(logits, loss_fn).squeeze()

        metrics_kwargs = {}
        if 'avg_feats_diff' in metrics:
            metrics_kwargs['X_valid'] = X_valid_tensor
            metrics_kwargs['train_features_mean'] = train_features_mean
        if 'f_beta_score' in metrics:
            metrics_kwargs['beta'] = kwargs['beta']
        metrics_values = compute_metrics(metrics, y_valid_tensor, y_valid_pred, **metrics_kwargs)

    return tuple(metrics_values.values())
