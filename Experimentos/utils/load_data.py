import numpy as np
import pandas as pd
import torch

def transform(individual_data, start, required_periods, is_control=None):
    periods = individual_data['t']
    periods = individual_data[periods.between(start-required_periods, start-1)]

    if len(periods) == required_periods:
        row = {
            'id': individual_data['id'].iloc[0],
            'inicio_prog': start,
            'tratado': individual_data['tratado'].iloc[0],
            'control': individual_data['control'].iloc[0] if is_control is None else is_control,
        }
        for i in range(required_periods):
            row[f"y(t-{required_periods-i})"] = periods['y'].values[i]
    else:
        raise ValueError(f"Individual {individual_data['id'].iloc[0]} does not have enough periods.")
    return row

def transform_treated(treated_data, required_periods):
    transformed_treated = []
    for _, individual in treated_data.groupby('id'):
        start = individual['inicio_prog'].iloc[0]
        row = transform(individual, start, required_periods)
        transformed_treated.append(row)
    return pd.DataFrame(transformed_treated)

def transform_control(control_data, min_start, max_start, required_periods):
    transformed_control = []
    for _, individual in control_data.groupby('id'):
        inicio_prog = individual['inicio_prog'].iloc[0]
        for assumed_start in range(min_start, max_start + 1):
            is_control = (1 if assumed_start == inicio_prog else 0)
            row = transform(individual, assumed_start, required_periods, is_control)
            transformed_control.append(row)
    return pd.DataFrame(transformed_control)

def transform_untreated(untreated_data, min_start, max_start, required_periods):
    transformed_untreated = []
    for _, individual in untreated_data.groupby('id'):
        for assumed_start in range(min_start, max_start + 1):
            row = transform(individual, assumed_start, required_periods)
            transformed_untreated.append(row)
    return pd.DataFrame(transformed_untreated)

def add_target_column(df):
    df['target'] = df['tratado'] | df['control']
    df.drop(columns=['tratado', 'control'], inplace=True)


def get_dfs(data, required_periods=4):
    type1_data = data[data['tipo'] == 1]
    type1_df = transform_treated(type1_data, required_periods)

    type2_data = data[data['tipo'] == 2]
    min_start = type1_df['inicio_prog'].min()
    max_start = type1_df['inicio_prog'].max()
    type2_df = transform_control(type2_data, min_start, max_start, required_periods)
    
    type3_data = data[data['tipo'] == 3]
    type3_df = transform_untreated(type3_data, min_start, max_start, required_periods)

    final_dfs = []
    for df in [type1_df, type2_df, type3_df]:
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


def build_train_valid_test_dfs(type1_df, type2_df, type3_df, partitions):
    """
    Builds the train, validation and test DataFrames with the amount of individuals
    of a certain type given by `partitions`.
    
    `partitions` is exepcted to be a dictionary like the following:
    {
        'train': {'type1': ..., 'type2': ..., 'type3': ...},
        'valid': {'type1': ..., 'type2': ..., 'type3': ...},
        'test' : {'type1': ..., 'type2': ..., 'type3': ...},
    }
    """
    if 'train' not in partitions or 'valid' not in partitions or 'test' not in partitions:
        raise ValueError("partitions must have 'train', 'valid' and 'test' keys.")
    
    def sample_rows(df, n):
        """Helper function to sample `n` rows from the DataFrame."""
        if n > len(df):
            raise ValueError(f"Cannot sample {n} rows from DataFrame with {len(df)} rows.")
        return df.sample(n=n, random_state=42)

    train_dfs = []
    valid_dfs = []
    test_dfs = []
    
    # Process each type
    for type_name, type_df in zip(['type1', 'type2', 'type3'], [type1_df, type2_df, type3_df]):
        for split, split_dfs in zip(['train', 'valid', 'test'], [train_dfs, valid_dfs, test_dfs]):
            num_samples = partitions[split][type_name]
            split_dfs.append(sample_rows(type_df, num_samples))
            # Drop sampled rows to avoid duplicates
            type_df = type_df.drop(split_dfs[-1].index)

    # Concatenate the DataFrames for each split
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    valid_df = pd.concat(valid_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    # Shuffle the final DataFrames
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Get X and y
    X_train_df, y_train_df = train_df.drop(columns=['target']), train_df['target']
    X_valid_df, y_valid_df = valid_df.drop(columns=['target']), valid_df['target']
    X_test_df, y_test_df = test_df.drop(columns=['target']), test_df['target']

    return (
        X_train_df, y_train_df, X_valid_df, y_valid_df, X_test_df, y_test_df
    )