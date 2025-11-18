import torch

from torch.utils.data import Dataset, TensorDataset
from utils.data_classes import TemporalStaticDataset
from utils.load_data import get_lstm_input

class ModelDatasetPreparer:
    def __init__(self, static_feats: list[str], temp_feats: list[str], model: str):
        """
        Build PyTorch-ready training and test datasets from temporal and static
        feature splits, adjusting tensor shapes for different model architectures.

        Args:
            static_feats (list[str]): Column names for static (non-temporal)
                features.
            temp_feats (list[str]): Column names for temporal features.
        """
        self.static_feats = static_feats
        self.temp_feats = temp_feats
        self.model = model

    def build_datasets(
        self, X_train_df, X_test_df, y_train_df, y_test_df
    ) -> tuple[Dataset, Dataset]:
        # labels should be of type float32 if using BCEWithLogitsLoss
        # labels should be of type long if using CrossEntropyLoss
        self.y_train_tensor = torch.tensor(y_train_df.values, dtype=torch.float32)
        self.y_test_tensor  = torch.tensor(y_test_df.values , dtype=torch.float32)

        if any(keyword in self.model for keyword in ['rnn', 'gru', 'lstm_v1']):
            train_set, test_set = self._prepare_lstm_v1(
                X_train_df, X_test_df, self.temp_feats, self.static_feats
            )

        elif 'dense' in self.model:
            train_set, test_set = self._prepare_dense(X_train_df, X_test_df)

        elif any(keyword in self.model for keyword in ['lstm_v2', 'bilstm']):
            train_set, test_set = self._prepare_lstm_v2(X_train_df, X_test_df)

        elif any(keyword in self.model for keyword in ['conv', 'lstm_conv']):
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
