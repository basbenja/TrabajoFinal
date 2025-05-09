import json
import os
import papermill as pm


NOTEBOOK_NAME = "main.ipynb"
NOTEBOOK_PATH = os.path.join(os.getcwd(), NOTEBOOK_NAME)

CONFIG_NAME = "config.json"
CONFIG_PATH = os.path.join(os.getcwd(), CONFIG_NAME)


def modify_config(config_path, new_params):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config.update(new_params)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


GROUPS_REQUIRED_PERIODS = {
    1: 45,
    2: 45
}

N_SIMULATIONS = 10
METRICS = ['f1_score', 'f_beta_score']
MODELS = ['lstm_v2', 'conv', 'dense']

for group, req_periods in GROUPS_REQUIRED_PERIODS.items():
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

    for metric in METRICS:
        modify_config(CONFIG_PATH, {"metrics": [metric]})
        if metric == 'f_beta_score':
            modify_config(CONFIG_PATH, {"beta": 0.5})
        
        for model in MODELS:
            modify_config(CONFIG_PATH, {"model_arch": model})

            for sim in range(1, N_SIMULATIONS+1):
                print(f"🔧 Modifying config for run {sim}")
                modify_config(CONFIG_PATH, {"simulation": sim})

                print(f"🚀 Running notebook for config: {sim}")
                pm.execute_notebook(
                    NOTEBOOK_PATH,
                    os.path.join(group_notebooks, f"output_Sim{sim}.ipynb"),
                    log_output=False
                )

                print(f"✅ Finished run {sim}\n")

    print(f"✅ Finished group {group}\n")