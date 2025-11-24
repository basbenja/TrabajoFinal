import os

from dotenv import load_dotenv
_ = load_dotenv(override=True)

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Relevant paths
THIS_DIR = os.path.join(BASE_DIR, "Experimentos")

DATA_DIR = os.getenv("DATA_DIR", os.path.join(THIS_DIR, "datasets"))
DATA_PARAMS_PATH = os.path.join(THIS_DIR, "data_params.json")

TRAIN_PARAMS_PATH = os.path.join(THIS_DIR, "train_params.json")

RESULTS_DIR = os.path.join(THIS_DIR, "results")

# optuna related
OPTUNA_SQLITE_PATH = os.getenv("OPTUNA_SQLITE_PATH", os.path.join(THIS_DIR, "optuna_studies.sqlite3"))
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_SQLITE_PATH}"

# mlflow related
HOST = "0.0.0.0"
PORT = 8080
TRACKING_SERVER_URI = f"http://{HOST}:{PORT}"

EXPERIMENT_PREFIX = "Control_Group_Identification"
EXPERIMENT_DESCRIPTION = "Control Group Identification with Neural Networks"
EXPERIMENT_TAGS = {
    "author": "bbas",
    "mlflow.note.content": EXPERIMENT_DESCRIPTION
}

# Stata related
STATA_PATH = "/usr/local/stata17"

# Model training related
N_EPOCHS = 100
N_LAYERS = 2
OPTIMIZER = "Adam"
DROPOUTS = [0.3, 0.5, 0.7]
HIDDEN_SIZES = [32, 64, 128]
BATCH_SIZES = [32, 64, 128]
LEARNING_RATES = [1e-4, 1e-3, 1e-2]

# Hyperparameter Optimization Parallelization
N_PROCESSES = 3
TRIALS_PER_PROCESS = 2
