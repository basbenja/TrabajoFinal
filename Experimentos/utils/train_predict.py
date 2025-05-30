import torch
import torch.nn as nn

from utils.metrics import compute_metrics

def train_step(model, dataloader, loss_fn, optimizer):
    """
    Train the model for one epoch and compute the average batch loss in the train set.
    """
    model.train()
    model_device = next(model.parameters()).device

    total_samples = 0
    total_loss = 0

    for batch in dataloader:
        *X, y = [x.to(model_device) for x in batch]
        curr_batch_size = len(y)

        logits = model(*X)

        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            loss = loss_fn(logits.view(-1), y)
        else:
            loss = loss_fn(logits, y)
        loss.backward()
        total_loss += loss.item() * curr_batch_size

        optimizer.step()
        optimizer.zero_grad()

        total_samples += curr_batch_size

    avg_loss = total_loss / total_samples
    return avg_loss


def validate_step(model, dataloader, loss_fn, metrics, **metrics_kwargs):
    """
    Validate the model after one epoch and compute the average batch loss and
    every metric in `metrics`.
    """
    model.eval()
    model_device = next(model.parameters()).device

    total_loss = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            *X, y = [x.to(model_device) for x in batch]
            curr_batch_size = len(y)

            logits = model(*X)

            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                loss = loss_fn(logits.view(-1), y)
            else:
                loss = loss_fn(logits, y)
            total_loss += loss.item() * curr_batch_size

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(predict(logits, loss_fn).cpu().numpy())

            total_samples += curr_batch_size

    avg_loss = total_loss / total_samples
    metrics_values = compute_metrics(
        metrics, all_labels, all_preds, **metrics_kwargs
    )

    return avg_loss, metrics_values


def predict(model_output, loss_fn, thresh=0.5):
    # The prediction depends on how was the model trained
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        y_pred = torch.argmax(model_output, dim=1)
    elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
        y_pred_probs = torch.sigmoid(model_output)
        y_pred = (y_pred_probs >= thresh).float()
    return y_pred