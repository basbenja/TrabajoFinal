import numpy as np
import pandas as pd
import torch

def transform(individual_data, start, required_periods):
    periods = individual_data['t']
    periods = individual_data[periods.between(start-required_periods, start-1)]

    if len(periods) == required_periods:
        row = {
            'id': individual_data['id'].iloc[0],
            'inicio_prog': start,
            'tratado': individual_data['tratado'].iloc[0],
            'control': individual_data['control'].iloc[0],
        }
        for i in range(required_periods):
            row[f"y(t-{required_periods-i})"] = periods['y'].values[i]
        return row

def transform_treated(treated_data, required_periods):
    transformed_treated = []
    for _, individual in treated_data.groupby('id'):
        start = individual['inicio_prog'].iloc[0]
        row = transform(individual, start, required_periods)
        if row:
            transformed_treated.append(row)
    return pd.DataFrame(transformed_treated)

def transform_untreated(untreated_data, min_start, max_start, required_periods):
    transformed_untreated = []
    for assumed_start in range(min_start, max_start + 1):
        for _, individual in untreated_data.groupby('id'):
            row = transform(individual, assumed_start, required_periods)
            if row:
                transformed_untreated.append(row)
    return pd.DataFrame(transformed_untreated)


def get_dfs(stata_path, required_periods=4):
    data = pd.read_stata(stata_path)
    
    type1_data = data[data['tratado'] == 1]
    type1_df = transform_treated(type1_data, required_periods)

    min_start = type1_df['inicio_prog'].min()
    max_start = type1_df['inicio_prog'].max()
    
    untreated_data = data[data['tratado'] == 0]
    untreated_df = transform_untreated(untreated_data, min_start, max_start, required_periods)
    type2_df = untreated_df[untreated_df['control'] == 1]
    type3_df = untreated_df[untreated_df['control'] == 0]
    
    return type1_df, type2_df, type3_df


def get_lstm_input(df, time_steps, num_features) -> torch.Tensor:
    n = len(df)
    data = np.zeros((n, time_steps, num_features))
    data[:, :, 0] = df['inicio_prog'].values.reshape(-1, 1)
    time_features = [f'y(t-{i})' for i in range(time_steps, 0, -1)]
    data[:, :, 1] = df[time_features].values
    data_tensor = torch.tensor(data, dtype=torch.float32)
    return data_tensor

def get_y(y_df, loss_fn) -> torch.Tensor:
    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
        y_tensor = torch.tensor(y_df, dtype=torch.long)
    elif isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        y_tensor = torch.tensor(y_df, dtype=torch.float)
    return y_tensor

def get_loaders(X_tensor, y_tensor, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return loader