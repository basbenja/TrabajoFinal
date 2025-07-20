import json
import os
import papermill as pm


NOTEBOOK_NAME = "psm.ipynb"
NOTEBOOK_PATH = os.path.join(os.getcwd(), NOTEBOOK_NAME)

CONFIG_NAME = "config.json"
CONFIG_PATH = os.path.join(os.getcwd(), CONFIG_NAME)


def modify_config(config_path, new_params):
    with open(config_path, 'r') as f:
        config = json.load(f)

    config.update(new_params)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

GROUP = 7
N_SIMULATIONS = 100
METRICS = ['f1_score', 'precision', 'recall']

modify_config(CONFIG_PATH, {"group": GROUP})
modify_config(CONFIG_PATH, {"metrics": METRICS})
modify_config(CONFIG_PATH, {"log_to_mlflow": "True"})

for sim in range(1, N_SIMULATIONS+1):
    print(f"🔧 Modifying config for run {sim}")
    modify_config(CONFIG_PATH, {"simulation": sim})

    pm.execute_notebook(
        input_path=NOTEBOOK_PATH,
        output_path=None,
        log_output=False
    )

    print(f"✅ Finished sim {sim}\n")
