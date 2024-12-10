import optuna
import pandas as pd

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
