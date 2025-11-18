import numpy as np
import pandas as pd
import torch

def transform(id, data, start, required_periods):
    relevant_periods = data['t'].between(start-required_periods, start-1)
    relevant_data = data[relevant_periods]

    # relevant_periods is a boolean series, that's why we can sum it
    if sum(relevant_periods) == required_periods:
        row = {
            'id': id,
            'inicio_prog': start,
            'tratado': relevant_data['tratado'].iloc[0],
            'control': relevant_data['control'].iloc[0],
        }
        for i in range(required_periods):
            row[f"y(t-{required_periods-i})"] = relevant_data['y'].values[i]
    else:
        raise ValueError(f"Individual {id} does not have enough periods.")
    return row


def transform_treated(treated_df, required_periods):
    transformed_treated = []
    for id, data in treated_df.groupby('id'):
        treatment_start = data['inicio_prog'].iloc[0]
        row = transform(id, data, treatment_start, required_periods)
        transformed_treated.append(row)
    return pd.DataFrame(transformed_treated)


def transform_control(control_df, min_start, max_start, required_periods):
    transformed_control = []
    for id, data in control_df.groupby('id'):
        for assumed_start in range(min_start, max_start + 1):
            row = transform(id, data, assumed_start, required_periods)
            transformed_control.append(row)
    return pd.DataFrame(transformed_control)


def transform_nini(nini_df, min_start, max_start, required_periods):
    transformed_nini = []
    for id, data in nini_df.groupby('id'):
        for assumed_start in range(min_start, max_start + 1):
            row = transform(id, data, assumed_start, required_periods)
            transformed_nini.append(row)
    return pd.DataFrame(transformed_nini)


def add_target_column(df):
    df['target'] = df['tratado'] | df['control']
    df.drop(columns=['tratado', 'control'], inplace=True)


def get_groups_dfs(df, required_periods=4):
    treated_df = df[df['tratado'] == 1]
    treated_df = transform_treated(treated_df, required_periods)

    min_start = treated_df['inicio_prog'].min()
    max_start = treated_df['inicio_prog'].max()
    control_df = df[df['control'] == 1]
    control_df = transform_control(control_df, min_start, max_start, required_periods)

    nini_df = df[(df['tratado'] == 0) & (df['control'] == 0)]
    nini_df = transform_nini(nini_df, min_start, max_start, required_periods)

    final_dfs = []
    for df in [treated_df, control_df, nini_df]:
        df = df.copy()
        df.set_index('id', inplace=True)
        add_target_column(df)
        final_dfs.append(df)

    return final_dfs


def get_lstm_input(df, time_feats, static_feats=None) -> torch.Tensor:
    """
    Transforms the features in the dataframe into a format compatible with the
    LSTM module. The LSTM module requires the input to be in format:
        (batch_size, seq_len, num_features)

    If we are using the static feature, the value of it replicates along the
    different time steps.
    """
    n = len(df)
    num_features = 1 if static_feats is None else 2
    data = np.zeros((n, len(time_feats), num_features))
    if not static_feats:
        data[:, :, 0] = df[time_feats].values
    else:
        data[:, :, 0] = df[time_feats].values
        data[:, :, 1] = df[static_feats].values.reshape(-1, 1)
    return torch.tensor(data, dtype=torch.float32)
