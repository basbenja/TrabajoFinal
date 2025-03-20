import optuna
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from utils.train_predict import train_step, validate_step_with_metrics
from utils.metrics import check_metrics, get_features_mean, compute_metrics
from sklearn.model_selection import StratifiedKFold

def objective(
    trial, define_model, train_set, valid_set, class_weights, metrics, **kwargs
):
    check_metrics(metrics, **kwargs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = define_model(trial).to(device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    epochs = trial.suggest_int("n_epochs", 100, 300)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(class_weights[1], dtype=torch.float32)
    )

    X_train, y_train = train_set.tensors
    if 'avg_feats_diff' in metrics:
        train_features_mean = get_features_mean(X_train, y_train).to(device)

    # Get the X_valid tensor so then we can calculate the features mean for the
    # individuals of the validation set that the model predicted as 1
    # And get the y_valid_tensor tensors so then we can calculate metrics
    X_valid, y_valid = valid_set.tensors
    X_valid = X_valid.to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        train_step(model, train_loader, loss_fn, optimizer)
        metrics_values = validate_step_with_metrics(
            model, X_valid, y_valid, loss_fn, metrics,
            train_features_mean=train_features_mean, beta=kwargs['beta']
        )

        if len(metrics) == 1:
            trial.report(metrics_values[metrics[0]], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return tuple(metrics_values.values())


def objective_cv(
    trial, define_model, train_set, class_weights, metrics, **kwargs
):
    check_metrics(metrics, **kwargs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = define_model(trial).to(device)

    # Hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    epochs = trial.suggest_int("n_epochs", 300, 1500)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(class_weights[1], dtype=torch.float32)
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=13)
    # Track metrics for all folds
    scores = {metric: [] for metric in metrics}
    X_train, y_train = train_set.tensors
    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        train_data = TensorDataset(X_train_fold, y_train_fold)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        X_valid_fold, y_valid_fold = X_train[valid_idx], y_train[valid_idx]
        X_valid_fold, y_valid_fold = X_valid_fold.to(device), y_valid_fold.to(device)

        if 'avg_feats_diff' in metrics:
            train_features_mean = get_features_mean(X_train_fold, y_train_fold).to(device)

        # Train model with the hyperparameters of the trial
        for _ in range(epochs):
            _ = train_step(model, train_loader, loss_fn, optimizer)

        metrics_values = validate_step_with_metrics(
            model, X_valid_fold, y_valid_fold, loss_fn, metrics,
            train_features_mean=train_features_mean if 'avg_feats_diff' in metrics else None,
            beta=kwargs['beta']
        )
        for metric, value in metrics_values.items():
            scores[metric].append(value)

    # Aggregate scores across folds
    aggregated_scores = {metric: np.mean(values) for metric, values in scores.items()}

    # if len(metrics) == 1:
    #     trial.report(metrics_values[metrics[0]], epoch)
    #     if trial.should_prune():
    #         raise optuna.TrialPruned()

    return tuple(aggregated_scores.values())


def get_all_studies(storage):
    # Get all study summaries
    study_summaries = optuna.get_all_study_summaries(storage)

    data = []
    for summary in study_summaries:
        study_name = summary.study_name
        start_date = summary.datetime_start.strftime("%Y-%m-%d %H:%M:%S")
        best_trial_id = summary.best_trial.number if summary.best_trial else None
        best_value = summary.best_trial.value if summary.best_trial else None
        data.append([study_name, start_date, best_trial_id, best_value])

    optuna_studies_df = pd.DataFrame(
        data,
        columns=["Study name", "Start date", "Best trial id", "Best objective value"]
    )
    return optuna_studies_df


def delete_studies(studies_names, storage):
    for name in studies_names:
        optuna.delete_study(name, storage)


def get_best_trials_info(study, metrics):
    best_trials = study.best_trials
    best_trials_list = []
    for trial in best_trials:
        trial_number = trial.number
        trial_params = trial.params
        trial_values = {}
        for metric, value in zip(metrics, trial.values):
            trial_values[metric] = value
        best_trials_list.append({
            "trial_number": trial_number,
            "params": trial_params,
            "values": trial_values
        })
    return best_trials_list


def select_trial(best_trials_numbers):
    while True:
        try:
            selected_trial = int(input(
                f"Enter the trial number. Choose from {best_trials_numbers}: "
            ))
            if selected_trial in best_trials_numbers:
                return selected_trial
            else:
                print(f"Invalid input. Please select a number from {best_trials_numbers}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
