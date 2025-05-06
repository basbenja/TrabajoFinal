import optuna
import pandas as pd
import torch
import numpy as np

from constants import N_EPOCHS, OPTIMIZER
from optuna.visualization import plot_pareto_front
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from utils.train_predict import train_step, validate_step
from utils.metrics import check_metrics


def objective_cv(
    trial, define_model, input_size, train_set, loss_fn, metrics, **kwargs
):
    check_metrics(metrics, **kwargs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = define_model(trial, input_size).to(device)

    lr = trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=lr)

    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=13)
    # Track metrics for all folds
    scores = {metric: [] for metric in metrics}

    labels = train_set.labels if hasattr(train_set, 'labels') else train_set.tensors[1]
    for train_idx, valid_idx in skf.split(np.arange(len(labels)), labels):
        train_subset = Subset(train_set, train_idx)
        valid_subset = Subset(train_set, valid_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)

        # Train model with the hyperparameters of the trial
        for _ in range(N_EPOCHS):
            _ = train_step(model, train_loader, loss_fn, optimizer)

        _, metrics_values = validate_step(
            model, valid_loader, loss_fn, metrics,
            train_features_mean=None, beta=kwargs['beta']
        )
        
        for metric, value in metrics_values.items():
            scores[metric].append(round(value, 6))

    # Aggregate scores across folds
    aggregated_scores = {
        metric: round(np.mean(values), 6) for metric, values in scores.items()
    }

    # if len(metrics) == 1:
    #     trial.report(metrics_values[metrics[0]], epoch)
    #     if trial.should_prune():
    #         raise optuna.TrialPruned()

    return tuple(aggregated_scores.values())


def get_all_studies(storage):
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


def get_study_by_id(study_id, storage):
    study = optuna.load_study(study_name=study_id, storage=storage)
    return study


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


def pareto_front(study, metrics, directions):
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
    for i, (metric, direction) in enumerate(zip(metrics, directions)):
        if direction == 'maximize':
            best_trial = max(study.best_trials, key=lambda t: t.values[i])
        elif direction == 'minimize':
            best_trial = min(study.best_trials, key=lambda t: t.values[i])
        print(f"Metric: {metric}")
        print(f"\tDirection: {direction}")
        print(f"\tTrial number: {best_trial.number}")
        print(f"\tValues: {best_trial.values}")
        print(f"\tParams: {best_trial.params}")
    
    fig = plot_pareto_front(study, target_names=metrics)
    return fig
