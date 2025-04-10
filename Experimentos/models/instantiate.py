from models.dense import DenseClassifier
from models.lstm_v1 import LSTMClassifier_v1
from models.lstm_v2 import LSTMClassifier_v2
from models.fcn import FCN_FC
from models.gru import GRUCLassifier

def instantiate_model(model_arch, input_size, hyperparams):
    dropout = hyperparams['dropout']
    match model_arch.lower():
        case 'dense':
            num_layers = hyperparams['num_layers']
            hidden_sizes = [hyperparams[f"n_units_l{i}"] for i in range(num_layers)]
            model = DenseClassifier(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                dropout=hidden_sizes,
            )
        case 'lstm_v1':
            num_layers = hyperparams['num_layers']
            hidden_size = hyperparams['hidden_size']
            model = LSTMClassifier_v1(
                input_size=input_size,
                num_layers=num_layers,
                hidden_size=hidden_size,
                dropout=dropout
            )
        case 'lstm_v2':
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
            num_layers = hyperparams['num_layers']
            hidden_size = hyperparams['hidden_size']
            model = GRUCLassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        case 'fcn':
            model = FCN_FC(
                n_static_feats=1
            )
    return model
