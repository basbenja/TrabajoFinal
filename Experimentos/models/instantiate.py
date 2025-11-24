import torch

from torch.utils.data import Dataset

from models import *
from models.blocks.ConvBlock import ConvBlock
from utils.data_classes import TemporalStaticDataset

# TODO: pensar cómo se puede mejorar todo esto. Creo que lo mejor sería que cada
# modelo tenga como su propia configuración.
# La función `define_...` se necesita para hacer la optimización de hiperparámetros
# con Optuna
# El input_size se necesita para definir el modelo (número de features de entrada)
# y varía según el modelo. Se necesita para la optimización y para la instancia final.


def instantiate_model(model_arch, input_size, hyperparams, **kwargs):
    match model_arch.lower():
        case 'dense':
            dropout = hyperparams['dropout']
            num_layers = hyperparams['num_layers']
            hidden_sizes = [hyperparams[f"n_units_l{i}"] for i in range(num_layers)]
            model = DenseClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                dropout=dropout,
            )
        case 'lstm_v1':
            dropout = hyperparams['dropout']
            num_layers = hyperparams['num_layers']
            hidden_size = hyperparams['hidden_size']
            model = LSTMClassifier_v1(
                input_size=input_size,
                num_layers=num_layers,
                hidden_size=hidden_size,
                dropout=dropout
            )
        case 'lstm_v2':
            dropout = hyperparams['dropout']
            num_layers = hyperparams['num_layers']
            hidden_size = hyperparams['hidden_size']
            model = LSTMClassifier_v2(
                lstm_input_size=1,
                lstm_hidden_size=hidden_size,
                lstm_num_layers=num_layers,
                n_static_feats=1,
                dropout=dropout
            )
        case 'gru':
            dropout = hyperparams['dropout']
            num_layers = hyperparams['num_layers']
            hidden_size = hyperparams['hidden_size']
            model = GRUCLassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        case 'conv':
            dropout = hyperparams['dropout']
            model = Conv_FC(
                dropout=dropout, n_static_feats=1, conv_out_dim=kwargs['conv_out_dim']
            )
        case 'lstm_conv':
            model = LSTMConvClassifier(
                lstm_input_size=input_size,
                lstm_hidden_size=hyperparams['hidden_size'],
                lstm_num_layers=hyperparams['num_layers'],
                n_static_feats=1,
                dropout=hyperparams['dropout'],
                conv_out_dim=kwargs['conv_out_dim']
            )
        case 'bilstm':
            model = BiLSTMClassifier(
                lstm_input_size=input_size,
                lstm_hidden_size=hyperparams['hidden_size'],
                lstm_num_layers=hyperparams['num_layers'],
                n_static_feats=1,
                dropout=hyperparams['dropout']
            )

    return model


def compute_conv_output_dim(train_set: Dataset) -> int:
    """Compute the output dimension of the conv block."""
    dummy_input = torch.zeros(
        (1, train_set.temporal_data.shape[1], train_set.temporal_data.shape[2])
    )
    conv_out_dim = ConvBlock(dropout=0)(dummy_input).shape[1]
    return conv_out_dim


def get_model_factory(model_arch: str, train_set: Dataset, feats: list[str]) -> tuple[callable, int]:
    if 'conv' in model_arch.lower() and not isinstance(train_set, TemporalStaticDataset):
        raise ValueError(f"Model architecture {model_arch} requires temporal and static features.")

    match model_arch.lower():
        case "lstm_v1":
            function = define_lstm_v1_model
            input_size = 2
        case "lstm_v2":
            function = define_lstm_v2_model
            input_size = 1
        case "gru":
            function = define_gru_model
            input_size = 2
        case "dense":
            function = define_dense_model
            input_size = len(feats)
        case "conv":
            conv_out_dim = compute_conv_output_dim(train_set)
            function = lambda trial, input_size: define_conv_model(trial, input_size, conv_out_dim)
            input_size = 1
        case "lstm_conv":
            conv_out_dim = compute_conv_output_dim(train_set)
            function = lambda trial, input_size: define_lstm_conv_model(trial, input_size, conv_out_dim)
            input_size = 1
        case "bilstm":
            function = define_bilstm_model
            input_size = 1

    return function, input_size
