import json
import os
import papermill as pm
import time

NOTEBOOK_NAME = "main.ipynb"
NOTEBOOK_PATH = os.path.join(os.getcwd(), NOTEBOOK_NAME)

CONFIG_NAME = "config.json"
CONFIG_PATH = os.path.join(os.getcwd(), CONFIG_NAME)

GROUPS = [1, 2, 3, 4]
REQUIRED_PERIODS = [50, 35, 25, 15]
N_SIMULATIONS = 10

MODELS = ["dense", "lstm_v1", "fcn"]

def modify_config(config_path, new_params):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config.update(new_params)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

for group, req_periods in zip(GROUPS, REQUIRED_PERIODS):
    print(f"🔧 Modifying config for group {group}")
    modify_config(
        CONFIG_PATH,
        {"group": group, "required_periods": req_periods}
    )
    
    group_notebooks = os.path.join(
        "/users/bbas/TrabajoFinal", "notebooks_outputs", f"notebooks_Grupo{group}"
    )
    os.makedirs(group_notebooks, exist_ok=True)
    print(f"📂 Created directory for group {group} notebooks")

    for sim in range(1, N_SIMULATIONS+1):
        print(f"🔧 Modifying config for run {sim}")
        modify_config(CONFIG_PATH, {"simulation": sim})

        print(f"🚀 Running notebook for config: {sim}")
        pm.execute_notebook(
            NOTEBOOK_PATH,
            os.path.join(group_notebooks, f"output_Sim{sim}.ipynb"),
            log_output=True
        )

        print(f"✅ Finished run {sim}\n")

    print(f"✅ Finished group {group}\n")