# optuna related
OPTUNA_STORAGE = "sqlite:////users/bbas/TrabajoFinal/optuna_db.sqlite3"

# mlflow related
HOST = "0.0.0.0"
PORT = 8080
TRACKING_SERVER_URI = f"http://{HOST}:{PORT}"

EXPERIMENT_NAME = "TrabajoFinal"
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
DATA_DIR = "/home/basbenja/Facultad/TrabajoFinal/data"

# Stata related
STATA_PATH = "/usr/local/stata17"

# Model training related
N_EPOCHS = 100
N_LAYERS = 2
OPTIMIZER = "Adam"

# Hyperparameter Optimization Parallelization
N_PROCESSES = 8
TRIALS_PER_PROCESS = 3
