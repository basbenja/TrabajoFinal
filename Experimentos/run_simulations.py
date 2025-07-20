import json
import os
import papermill as pm
import time

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


today = time.strftime("%Y%m%d")

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

def create_sim_dir(base_path, timestamp):
    sim_dir_path = os.path.join(base_path, f"simulations_{timestamp}")
    os.makedirs(sim_dir_path, exist_ok=True)
    return sim_dir_path


GROUPS = [7]
N_SIMULATIONS = 100
METRICS = ['f1_score']
MODELS = ['lstm_v2', 'conv', 'lstm_conv']

sim_dir_path = create_sim_dir("/users/bbas/TrabajoFinal", today)
modify_config(CONFIG_PATH, {"log_to_mlflow": "True", "scale_data": "False"})

for group in GROUPS:
    logger.info(f"🔧 Modifying config for group {group}")
    modify_config(CONFIG_PATH, {"group": group})
    
    group_notebooks = os.path.join(sim_dir_path, f"notebooks_Grupo{group}")
    os.makedirs(group_notebooks, exist_ok=True)
    logger.info(f"📂 Created notebooks directory for group {group} notebooks")

    for metric in METRICS:
        modify_config(CONFIG_PATH, {"metrics": [metric]})
        if metric == 'f_beta_score':
            modify_config(CONFIG_PATH, {"beta": 0.5})
        
        for model in MODELS:
            logger.info(f"🔧 Modifying config for model arch {model}")
            modify_config(CONFIG_PATH, {"model_arch": model})

            for sim in range(1, N_SIMULATIONS+1):
                logger.info(f"🔧 Modifying config for run {sim}")
                modify_config(CONFIG_PATH, {"simulation": sim})

                logger.info(f"🚀 Running notebook for config: {sim}")
                pm.execute_notebook(
                    NOTEBOOK_PATH,
                    os.path.join(group_notebooks, f"output_Sim{sim}.ipynb"),
                    log_output=False
                )

                logger.info(f"✅ Finished run {sim}\n")

    logger.info(f"✅ Finished group {group}\n")
