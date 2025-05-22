# optuna related
OPTUNA_STORAGE = "sqlite:////users/bbas/TrabajoFinal/optuna_db.sqlite3"

# mlflow related
HOST = "0.0.0.0"
PORT = 8080
TRACKING_SERVER_URI = f"http://{HOST}:{PORT}"

EXPERIMENT_PREFIX = "TF"
EXPERIMENT_DESCRIPTION = (
    "Final Project of Computer Science MS at FAMAF-UNC: Control Group Identification "
    "with Neural Networks"
)
EXPERIMENT_TAGS = {
    "project_name": "Trabajo Final",
    "author": "bbas",
    "mlflow.note.content": EXPERIMENT_DESCRIPTION
}

# Simulated databases related
DATA_DIR = "/users/bbas/TrabajoFinal/data/Benja"

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
N_PROCESSES = 8
TRIALS_PER_PROCESS = 2
