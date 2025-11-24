import json

from pprint import pp

from training import Trainer
from constants import TRAIN_PARAMS_PATH

if __name__ == "__main__":
    # 1. Read params
    with open(TRAIN_PARAMS_PATH, 'r') as f:
        params = json.load(f)
    print("Parameters for training:")
    pp(params)

    # 2. Create trainer (fast, safe)
    trainer = Trainer(params)
    print(
        f"Training model {trainer.model_arch} for group: {trainer.group}, "
        f"simulation: {trainer.simulation}\n"
    )

    # 3. Load and prepare data (explicit steps)
    trainer.load_data()
    trainer.prepare_train_test_split()

    # 4. Get datasets
    train_set, test_set = trainer.get_datasets()
