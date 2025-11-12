import json
import os
import pandas as pd

from constants import DATA_DIR, TRACKING_SERVER_URI, EXPERIMENT_PREFIX
from logger import MLflowLogger

class Trainer:
    def __init__(self, params: dict) -> "Trainer":
        self.params = params

        self._parse_params()
        self._set_up_logger()
        self._load_group_params()

        self.df = pd.read_stata(self.dataset_path)

        self._log_params()

    def _parse_params(self) -> None:
        """
        Parse the parameters dictionary and set instance variables accordingly.
        """
        p = self.params

        self.group = f"group_{p["group"]}"
        self.group_path = os.path.join(DATA_DIR, self.group)
        if not os.path.exists(self.group_path):
            raise ValueError(f"Group path {self.group_path} does not exist.")

        self.simulation = p["simulation"]
        self.dataset_path = os.path.join(self.group_path, f"simulation_{self.simulation}.dta")
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {self.dataset_path} does not exist.")

        self.model_arch = p["model_arch"]
        self.metrics: list[str] = p["metrics"]
        self.directions: list[str] = p["directions"]
        if len(self.metrics) != len(self.directions):
            raise ValueError("Length of metrics and directions must be the same.")

        self.beta = p["beta"]

        self.log_to_mlflow = (p['log_to_mlflow'] == "True")
        self.scale_data = (p['scale_data'] == "True")

    def _set_up_logger(self) -> None:
        """
        Set up the MLflow logger if logging is enabled.
        """
        # NOTA: por más que no querramos loguear a MLflow, igual instanciamos
        # el logger porque tenemos que loguear en el medio del código. Es mejor
        # hacer esto que poner condicionales por todos lados.
        self.mlflow_logger = MLflowLogger(
            enable_logging=self.log_to_mlflow,
            tracking_server_uri=TRACKING_SERVER_URI,
            experiment_name=f"{EXPERIMENT_PREFIX}-{self.group}",
        )

    def _load_group_params(self) -> None:
        """
        Log the group-level parameters to MLflow.
        """
        group_params_file = os.path.join(self.group_path, f"params.json")
        with open(group_params_file, 'r') as f:
            self.group_params = json.load(f)

        self.req_periods = self.group_params['first_tr_period'] - 1
        self.temp_feats = [f'y(t-{i})' for i in range(self.req_periods, 0, -1)]
        self.stat_feats = ['inicio_prog']
        self.feats = self.stat_feats + self.temp_feats
        self.n_per_dep = self.group_params['n_per_dep']

    def _log_params(self) -> None:
        self.mlflow_logger.log_params({
            "group": self.group,
            "simulation": self.simulation,
            "filepath": self.dataset_path,
            "required_periods": self.req_periods,
            "n_per_dep": self.n_per_dep,
            "scale_data": self.scale_data,
            "model_arch": self.model_arch,
            "metrics": self.metrics,
            "ups_max_count": self.group_params['ups_max_count']
        })

        if 'f_beta_score' in self.metrics:
            self.mlflow_logger.log_param("beta", self.beta)