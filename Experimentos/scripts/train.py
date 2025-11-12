import json
import os

from pprint import pp

from training import Trainer
from constants import TRAIN_PARAMS_PATH

if __name__ == "__main__":
    # 1. Read the params for the training process
    with open(TRAIN_PARAMS_PATH, 'r') as f:
        params = json.load(f)
    print("Parameters for training:")
    pp(params)
    print()

    # 2. Instatiate a Trainer
    trainer = Trainer(params)
    print(f"Training model for group: {trainer.group}, simulation: {trainer.simulation}\n")

    # 3. ...
    # 4. ...
