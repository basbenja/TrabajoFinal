import torch
import torch.nn as nn

from utils.metrics import compute_metrics

def train_step(model, dataloader, loss_fn, optimizer):
    """
    Train the model for one epoch and compute the average batch loss in the train set.
    """
    num_batches = len(dataloader)
    total_loss = 0
    model_device = next(model.parameters()).device

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(model_device), y.to(model_device)
        logits = model(X)

        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            loss = loss_fn(logits.view(-1), y)
        else:
            loss = loss_fn(logits, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_step(model, dataloader, loss_fn):
    """
    Validate the model after one epoch and compute the average batch loss and
    every metric in `metrics`.
    """
    num_batches = len(dataloader)
    total_loss = 0
    model_device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(model_device), y.to(model_device)
            logits = model(X)

            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                loss = loss_fn(logits.view(-1), y)
            else:
                loss = loss_fn(logits, y)

            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_step_with_metrics(
    model, X, y, loss_fn, metrics, **kwargs
):
    """
    Validate the model after one epoch. Compute every metric in `metrics`.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_valid_pred = predict(logits, loss_fn).squeeze()

    metrics_kwargs = {}
    if 'avg_feats_diff' in metrics:
        metrics_kwargs['X_valid'] = X
        metrics_kwargs['train_features_mean'] = kwargs['train_features_mean']
    if 'f_beta_score' in metrics:
        metrics_kwargs['beta'] = kwargs['beta']
    metrics_values = compute_metrics(
        metrics, y, y_valid_pred, **metrics_kwargs
    )
    return metrics_values


def predict(model_output, loss_fn):
    # The prediction depends on how was the model trained
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        y_pred = torch.argmax(model_output, dim=1)
    elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
        y_pred_probs = torch.sigmoid(model_output)
        y_pred = (y_pred_probs >= 0.5).float()
    return y_pred