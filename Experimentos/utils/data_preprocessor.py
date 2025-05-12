import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from utils.data_classes import TemporalStaticDataset
from utils.load_data import get_lstm_input

class DataPreprocessor:
    def __init__(self, static_feats, temp_feats):
        self.static_feats = static_feats
        self.temp_feats = temp_feats

    def _scale_static_feats(self, train_df, test_df):
        scaler = StandardScaler()

        train_df_scaled = train_df.copy()
        train_df_scaled[self.static_feats] = scaler.fit_transform(train_df[self.static_feats])

        test_df_scaled = test_df.copy()
        test_df_scaled[self.static_feats] = scaler.transform(test_df[self.static_feats])

        return train_df_scaled, test_df_scaled

    def _scale_temp_feats(self, df):
        df_temp = df[self.temp_feats]
        
        df_temp_np = df_temp.to_numpy()
        means = df_temp_np.mean(axis=1, keepdims=True)
        stds  = df_temp_np.std (axis=1, keepdims=True)
        df_temp_np_standarized = (df_temp_np - means) / stds
        
        df_scaled = df.copy()
        df_scaled[self.temp_feats] = df_temp_np_standarized
        return df_scaled

    def _scale_data(self, train_df, test_df):
        # FIXME
        train_df_scaled = self._scale_temp_feats(train_df)
        test_df_scaled  = self._scale_temp_feats(test_df)
        
        train_df_scaled, test_df_scaled = self._scale_static_feats(
            train_df_scaled, test_df_scaled
        )
        return train_df_scaled, test_df_scaled
    
    def build_datasets(
        self, X_train_df, X_test_df, y_train_df, y_test_df, model_arch
    ):
        # labels should be of type float32 if using BCEWithLogitsLoss
        # labels should be of type long if using CrossEntropyLoss
        self.y_train_tensor = torch.tensor(y_train_df.values, dtype=torch.float32)
        self.y_test_tensor  = torch.tensor(y_test_df.values , dtype=torch.float32)

        if any(keyword in model_arch for keyword in ['rnn', 'gru', 'lstm_v1']):
            train_set, test_set = self._prepare_lstm_v1(
                X_train_df, X_test_df, self.temp_feats, self.static_feats
            )
        elif 'dense' in model_arch:
            train_set, test_set = self._prepare_dense(X_train_df, X_test_df)
        elif any(keyword in model_arch for keyword in ['lstm_v2', 'bilstm']):
            train_set, test_set = self._prepare_lstm_v2(X_train_df, X_test_df)
        elif any(keyword in model_arch for keyword in ['conv', 'lstm_conv']):
            train_set, test_set = self._prepare_conv(X_train_df, X_test_df)
        return train_set, test_set

    def _prepare_lstm_v1(self, X_train_df, X_test_df, temp_feats, stat_feat):
        X_train_tensor = get_lstm_input(X_train_df, temp_feats, stat_feat)
        X_test_tensor  = get_lstm_input(X_test_df , temp_feats, stat_feat)
        train_set = TensorDataset(X_train_tensor, self.y_train_tensor)
        test_set  = TensorDataset(X_test_tensor , self.y_test_tensor)
        return train_set, test_set

    def _prepare_dense(self, X_train_df, X_test_df):
        X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)
        X_test_tensor  = torch.tensor(X_test_df.values, dtype=torch.float32)
        train_set = TensorDataset(X_train_tensor, self.y_train_tensor)
        test_set  = TensorDataset(X_test_tensor , self.y_test_tensor)
        return train_set, test_set

    def _prepare_lstm_v2(self, X_train_df, X_test_df):
        X_train_temp_tensor = get_lstm_input(X_train_df, self.temp_feats)
        X_test_temp_tensor  = get_lstm_input(X_test_df , self.temp_feats)
        X_train_static_tensor = torch.tensor(
            X_train_df[self.static_feats].values, dtype=torch.float32
        ).view(-1, 1)
        X_test_static_tensor  = torch.tensor(
            X_test_df[self.static_feats].values , dtype=torch.float32
        ).view(-1, 1)
        train_set = TemporalStaticDataset(
            X_train_temp_tensor, X_train_static_tensor, self.y_train_tensor
        )
        test_set = TemporalStaticDataset(
            X_test_temp_tensor, X_test_static_tensor, self.y_test_tensor
        )
        return train_set, test_set

    def _prepare_conv(self, X_train_df, X_test_df):
        X_train_temp_tensor = torch.tensor(
            X_train_df[self.temp_feats].values, dtype=torch.float32
        ).unsqueeze(1)
        X_test_temp_tensor  = torch.tensor(
            X_test_df[self.temp_feats].values , dtype=torch.float32
        ).unsqueeze(1)
        X_train_static_tensor = torch.tensor(
            X_train_df[self.static_feats].values, dtype=torch.float32
        ).view(-1, 1)
        X_test_static_tensor  = torch.tensor(
            X_test_df[self.static_feats].values , dtype=torch.float32
        ).view(-1, 1)
        train_set = TemporalStaticDataset(
            X_train_temp_tensor, X_train_static_tensor, self.y_train_tensor
        )
        test_set = TemporalStaticDataset(
            X_test_temp_tensor, X_test_static_tensor, self.y_test_tensor
        )
        return train_set, test_set