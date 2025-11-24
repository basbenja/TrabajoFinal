import json

from pprint import pp

from models.instantiate import get_model_factory
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

    # 5. Get input size and model factory
    model_factory, input_size = get_model_factory(
        trainer.model_arch, train_set, trainer.feats
    )
    print(f"Model factory: {model_factory}")
    print(f"Input size: {input_size}\n")

    # 5. Optimize
    print("Starting hyperparameter optimization...")
    best_hyperparams = trainer.optimize(train_set, model_factory, input_size)
