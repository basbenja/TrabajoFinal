import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from utils.data_classes import TemporalStaticDataset
from utils.load_data import get_lstm_input

class DataPreprocessor:
    def __init__(self, X_train_df, X_test_df, y_train_df, y_test_df):
        self.X_train_df = X_train_df
        self.X_test_df = X_test_df
        self.y_train_df = y_train_df
        self.y_test_df = y_test_df

    def _scale_data(self):
        scaler = StandardScaler().fit(self.X_train_df)
        X_train_scaled = scaler.transform(self.X_train_df)
        self.X_train_df_scaled = pd.DataFrame(
            X_train_scaled, columns=self.X_train_df.columns, index=self.X_train_df.index
        )
        X_test_scaled = scaler.transform(self.X_test_df)
        self.X_test_df_scaled = pd.DataFrame(
            X_test_scaled, columns=self.X_test_df.columns, index=self.X_test_df.index
        )
        return self.X_train_df_scaled, self.X_test_df_scaled
    
    def build_datasets(self, model_arch, temp_feats, stat_feat, scale=True):
        if scale:
            X_train_df, X_test_df = self._scale_data()
        else:
            X_train_df, X_test_df = self.X_train_df, self.X_test_df

        # labels should be of type float32 if using BCEWithLogitsLoss
        # labels should be of type long if using CrossEntropyLoss
        self.y_train_tensor = torch.tensor(self.y_train_df.values, dtype=torch.float32)
        self.y_test_tensor  = torch.tensor(self.y_test_df.values , dtype=torch.float32)

        if any(keyword in model_arch for keyword in ['rnn', 'gru', 'lstm_v1']):
            train_set, test_set = self._prepare_lstm_v1(
                X_train_df, X_test_df, temp_feats, stat_feat
            )
        elif 'dense' in model_arch:
            train_set, test_set = self._prepare_dense(X_train_df, X_test_df)
        elif any(keyword in model_arch for keyword in ['lstm_v2', 'fcn']):
            train_set, test_set = self._prepare_lstm_v2_fcn(
                X_train_df, X_test_df, temp_feats, stat_feat
            )
        return train_set, test_set

    def _prepare_lstm_v1(self, X_train_df, X_test_df, temp_feats, stat_feat):
        X_train_tensor = get_lstm_input(X_train_df, temp_feats, stat_feat)
        X_test_tensor  = get_lstm_input(X_test_df , temp_feats, stat_feat)
        train_set = TensorDataset(X_train_tensor, self.y_train_tensor)
        test_set  = TensorDataset(X_test_tensor , self.y_test_tensor)
        return train_set, test_set

    def _prepare_dense(self, X_train_df, X_test_df):
        X_train_tensor = torch.tensor(X_train_df, dtype=torch.float32)
        X_test_tensor  = torch.tensor(X_test_df, dtype=torch.float32)
        train_set = TensorDataset(X_train_tensor, self.y_train_tensor)
        test_set  = TensorDataset(X_test_tensor , self.y_test_tensor)
        return train_set, test_set

    def _prepare_lstm_v2_fcn(self, X_train_df, X_test_df, temp_feats, stat_feat):
        X_train_temp_tensor = get_lstm_input(X_train_df, temp_feats)
        X_test_temp_tensor  = get_lstm_input(X_test_df , temp_feats)
        X_train_static_tensor = torch.tensor(
            self.X_train_df[stat_feat].values, dtype=torch.float32
        ).view(-1, 1)
        X_test_static_tensor  = torch.tensor(
            self.X_test_df[stat_feat].values , dtype=torch.float32
        ).view(-1, 1)
        train_set = TemporalStaticDataset(
            X_train_temp_tensor, X_train_static_tensor, self.y_train_tensor
        )
        test_set = TemporalStaticDataset(
            X_test_temp_tensor, X_test_static_tensor, self.y_test_tensor
        )
        return train_set, test_set