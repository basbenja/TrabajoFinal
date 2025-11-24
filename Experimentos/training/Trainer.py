import json
import numpy as np
import os
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

from constants import DATA_DIR, TRACKING_SERVER_URI, EXPERIMENT_PREFIX
from data import ModelDatasetPreparer
from logger import MLflowLogger
from models import *
from .Optimizer import Optimizer
from utils.load_data import get_groups_dfs

class Trainer:
    def __init__(self, params: dict):
        """Lightweight initialization - validation and setup only."""
        self.params = params

        self._parse_params()
        self._validate_paths()

        self._set_up_logger()

        self._load_group_params()

    def _validate_paths(self) -> None:
        """Validate paths exist without loading data."""
        if not os.path.exists(self.group_path):
            raise ValueError(f"Group path {self.group_path} does not exist.")
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Dataset path {self.dataset_path} does not exist.")

    def load_data(self) -> None:
        """Explicitly load the dataset when needed."""
        if hasattr(self, 'df') and self.df is not None:
            return  # Already loaded

        print(f"Loading data from {self.dataset_path}...")
        self.df = pd.read_stata(self.dataset_path)
        print(f"Loaded {len(self.df)} rows")

    def _parse_params(self) -> None:
        """
        Parse the parameters dictionary and set instance variables accordingly.
        """
        p = self.params

        self.group = f"group_{p["group"]}"
        self.group_path = os.path.join(DATA_DIR, self.group)

        self.simulation = p["simulation"]
        self.dataset_path = os.path.join(self.group_path, f"simulation_{self.simulation}.dta")

        self.model_arch = p["model_arch"]
        self.metrics: list[str] = p["metrics"]
        self.beta = p["beta"]

        self.log_to_mlflow = (p['log_to_mlflow'] == "True")

        self.n_type3_train = p["n_type3_train"]
        self.n_type3_test = p["n_type3_test"]

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
            "model_arch": self.model_arch,
            "metrics": self.metrics,
            "ups_max_count": self.group_params['ups_max_count']
        })

        if 'f_beta_score' in self.metrics:
            self.mlflow_logger.log_param("beta", self.beta)

    def prepare_train_test_split(self) -> None:
        if not hasattr(self, 'df') or self.df is None:
            self.load_data()

        self._log_params()

        type1_df, type2_df, type3_df = get_groups_dfs(self.df, self.req_periods)

        type3_ids = type3_df.index.unique()
        type3_train_ids = np.random.choice(type3_ids, self.n_type3_train, replace=False)
        type3_train_df = type3_df.loc[type3_train_ids]

        # Los ids que no están en type3_train son para el conjunto de testeo
        type3_test_ids = list(set(type3_ids) - set(type3_train_ids))
        type3_test_ids = np.random.choice(type3_test_ids, self.n_type3_test, replace=False)
        type3_test_df = type3_df.loc[type3_test_ids]

        self.train_df = pd.concat([type1_df, type3_train_df])
        self.X_train_df, self.y_train_df = self.train_df[self.feats], self.train_df['target']

        self.test_df = pd.concat([type2_df, type3_test_df])
        self.X_test_df, self.y_test_df = self.test_df[self.feats], self.test_df['target']

        self.weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.y_train_df),
            y=self.y_train_df
        )

    def get_datasets(self) -> tuple[Dataset, Dataset]:
        """Build PyTorch datasets from prepared data."""
        preparer = ModelDatasetPreparer(
            static_feats=self.stat_feats,
            temp_feats=self.temp_feats,
            model=self.model_arch
        )

        train_set, test_set = preparer.build_datasets(
            self.X_train_df,
            self.X_test_df,
            self.y_train_df,
            self.y_test_df
        )

        return train_set, test_set

    def optimize(self, train_set: Dataset, model_factory: callable, input_size: int) -> dict:
        """Run hyperparameter optimization."""
        optimizer = Optimizer(
            model_factory=model_factory,
            input_size=input_size,
            train_set=train_set,
            metrics=self.metrics,
            weights=self.weights,
            beta=self.beta,
        )

        return optimizer.run()
