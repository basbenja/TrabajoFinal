from models.dense import DenseClassifier
from models.lstm_v1 import LSTMClassifier_v1
from models.lstm_v2 import LSTMClassifier_v2
from models.fcn import FCN_FC
from models.gru import GRUCLassifier

def instantiate_model(model_arch, input_size, hyperparams):
    dropout = hyperparams['dropout']
    match model_arch.lower():
        case 'dense':
            n_layers = hyperparams['n_layers']
            hidden_sizes = [hyperparams[f"n_units_l{i}"] for i in range(n_layers)]
            model = DenseClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                dropout=hidden_sizes,
            )
        case 'lstm_v1':
            n_layers = hyperparams['n_layers']
            hidden_size = hyperparams['hidden_size']
            model = LSTMClassifier_v1(
                input_size=input_size,
                n_layers=n_layers,
                hidden_size=hidden_size,
                dropout=dropout
            )
        case 'lstm_v2':
            n_layers = hyperparams['n_layers']
            hidden_size = hyperparams['hidden_size']
            model = LSTMClassifier_v2(
                lstm_input_size=1,
                lstm_hidden_size=hidden_size,
                lstm_n_layers=n_layers,
                n_static_feats=1,
                dropout=dropout
            )
        case 'gru':
            n_layers = hyperparams['n_layers']
            hidden_size = hyperparams['hidden_size']
            model = GRUCLassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                n_layers=n_layers,
                dropout=dropout
            )
        case 'fcn':
            model = FCN_FC(
                n_static_feats=1
            )
    return model
