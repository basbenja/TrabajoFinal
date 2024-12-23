# optuna related
OPTUNA_STORAGE = "sqlite:///optuna_db.sqlite3"

# mlflow related
HOST = "0.0.0.0"
PORT = 8080
TRACKING_SERVER_URI = f"http://{HOST}:{PORT}"

EXPERIMENT_NAME = "Final Project - CS - FAMAF - 2024"
EXPERIMENT_DESCRIPTION = (
    "Final Project of Computer Science MS at FAMAF-UNC: Control Group Identification "
    "with Neural Networks"
)
EXPERIMENT_TAGS = {
    "project_name": "FinalProject-CS-FAMAF-2024",
    "author": "bbas",
    "mlflow.note.content": EXPERIMENT_DESCRIPTION
}

# Simulated databases related
DATA_DIR = "/users/bbas/TrabajoFinal/databases"
