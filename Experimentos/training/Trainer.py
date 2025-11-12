import json
import numpy as np
import os
import pandas as pd

from utils.load_data import get_dfs
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

    def _split_train_test(self):
        type1_df, type2_df, type3_df = get_dfs(self.df, self.req_periods)

        type1_ids = type1_df.index.unique()
        n_type1_train = 1000
        type1_train_ids = np.random.choice(type1_ids, n_type1_train, replace=False)
        type1_train_df = type1_df.loc[type1_train_ids]

        type3_ids = type3_df.index.unique()
        n_type3_train = 1000
        type3_train_ids = np.random.choice(type3_ids, n_type3_train, replace=False)
        type3_train_df = type3_df.loc[type3_train_ids]

        # Los ids que no están en type3_train son para el conjunto de testeo
        n_type3_test = 2500
        type3_test_ids = list(set(type3_ids) - set(type3_train_ids))
        type3_test_ids = np.random.choice(type3_test_ids, n_type3_test, replace=False)
        type3_test_df = type3_df.loc[type3_test_ids]

        self.train_df = pd.concat([type1_train_df, type3_train_df])
        self.X_train_df, self.y_train_df = self.train_df[self.feats], self.train_df['target']

        self.test_df = pd.concat([type2_df, type3_test_df])
        self.X_test_df, self.y_test_df = self.test_df[self.feats], self.test_df['target']

        return self.X_train_df, self.y_train_df, self.X_test_df, self.y_test_df