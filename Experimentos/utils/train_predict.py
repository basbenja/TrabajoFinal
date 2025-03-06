import torch
import torch.nn as nn

from tqdm.notebook import tqdm
from utils.early_stopping import EarlyStopping
from utils.metrics import compute_metrics

def train_step(model, dataloader, loss_fn, optimizer):
    """
    Train the model for one epoch.
    """
    num_batches = len(dataloader)
    total_loss = 0
    model.train()
    model_device = next(model.parameters()).device
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(model_device), y.to(model_device)
        y_pred = model(X)
        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            loss = loss_fn(y_pred.view(-1), y)
        else:
            loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # total_loss += loss.item()

    # avg_loss = total_loss / num_batches
    # print(f"Train loss: {avg_loss:.4f}")

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



def validate_step(model, dataloader, loss_fn):
    """
    Validate the model after one epoch. Compute loss and accuracy.
    """
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    total_loss, total_accuracy = 0, 0

    model.eval()
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(model_device), y.to(model_device)
            y_pred = model(X)

            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                loss = loss_fn(y_pred.view(-1), y)
            else:
                loss = loss_fn(y_pred, y)
            total_loss += loss.item()

            y_pred = predict(y_pred, loss_fn).squeeze()
            total_accuracy += (y_pred == y).type(torch.float).sum().item()

    avg_loss = total_loss / num_batches
    avg_acc = total_accuracy / size

    return avg_acc, avg_loss


def train_validate_loop(
    model, train_dataloader, test_dataloader, optimizer, loss_fn, num_epochs
):
    y_train_accs, train_losses = [], []
    y_test_accs , test_losses  = [], []

    for epoch in tqdm(range(1, num_epochs+1)):
        tqdm.write(f"Epoch {epoch}")
        train_step(model, train_dataloader, loss_fn, optimizer)

        train_acc, train_loss = validate_step(model, train_dataloader, loss_fn)
        test_acc , test_loss  = validate_step(model, test_dataloader , loss_fn)

        tqdm.write(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}")
        tqdm.write(f"Test loss : {test_loss:.4f} | Test accuracy : {test_acc:.4f}")
        tqdm.write("----------------------------------------------------------------")

        y_train_accs.append(train_acc)
        train_losses.append(train_loss)
        y_test_accs.append(test_acc)
        test_losses.append(test_loss)

    print("Training finished!")
    return model, y_train_accs, train_losses, y_test_accs, test_losses


def predict(model_output, loss_fn):
    # The prediction depends on how was the model trained
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        y_pred = torch.argmax(model_output, dim=1)
    elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
        y_pred_probs = torch.sigmoid(model_output)
        y_pred = (y_pred_probs >= 0.5).float()
    return y_pred