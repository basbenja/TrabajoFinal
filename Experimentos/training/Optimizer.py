import time
import optuna
import torch.nn as nn
import warnings
# Suppress Optuna experimental warnings
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

from joblib import Parallel, delayed
from torch.utils.data import Dataset

from utils.optuna_utils import *
from constants import (
    OPTUNA_STORAGE,
    N_PROCESSES,
    TRIALS_PER_PROCESS
)

class Optimizer:
    """Manages Optuna hyperparameter optimization with parallel trials."""
    def __init__(
        self,
        model_factory: callable,
        input_size: int,
        train_set: Dataset,
        metrics: dict[str, str],
        weights: np.ndarray,
        beta: float = 1.0,
    ):
        self.model_factory = model_factory
        self.input_size = input_size
        self.train_set = train_set
        self.metrics = metrics
        self.weights = weights
        self.beta = beta

    def _create_study(self) -> None:
        """Create Optuna study with appropriate direction."""
        timestamp = time.strftime("%d%m%Y-%H%M%S")
        self.study_name = f"study_{timestamp}"

        if len(self.metrics) == 1:
            direction = list(self.metrics.values())[0]
            directions = None
        else:
            direction = None
            directions = list(self.metrics.values())

        self.study = optuna.create_study(
            direction=direction,
            directions=directions,
            storage=OPTUNA_STORAGE,
            study_name=self.study_name,
            load_if_exists=True
        )
        self.study.set_metric_names(list(self.metrics.keys()))

    def _run_worker(self, n_trials: int) -> None:
        """Single worker function for parallel optimization."""
        study = optuna.load_study(
            study_name=self.study_name,
            storage=OPTUNA_STORAGE
        )

        # Por ahora, vamos a usar siempre BCEWithLogitsLoss ya que se trata de
        # un problema de clasificación binaria
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.weights[1], dtype=torch.float32)
        )

        study.optimize(
            lambda trial: objective_cv(
                trial,
                define_model=self.model_factory,
                input_size=self.input_size,
                train_set=self.train_set,
                loss_fn=loss_fn,
                metrics=list(self.metrics.keys()),
                beta=self.beta
            ),
            n_trials=n_trials,
            n_jobs=-1,
        )

    def _run_parallel_optimization(self, n_trials: int) -> None:
        """Run optimization in parallel."""
        trials_per_worker = n_trials // N_PROCESSES

        Parallel(n_jobs=N_PROCESSES)(
            delayed(self._run_worker)(trials_per_worker)
            for _ in range(N_PROCESSES)
        )

    def _get_best_params(self) -> dict:
        """Extract best hyperparameters from completed study."""
        return self.study.best_params

    def run(self, n_trials: int = None) -> dict:
        """Run optimization and return best hyperparameters."""
        n_trials = n_trials or (N_PROCESSES * TRIALS_PER_PROCESS)

        self._create_study()
        self._run_parallel_optimization(n_trials)

        # return self._get_best_params()


    # best_trials_info = get_best_trials_info(study, METRICS)
    # best_trials_numbers = [trial['trial_number'] for trial in best_trials_info]
    # pprint.pp(f"Best trials info: {best_trials_info}")
    # pprint.pp(f"Best trials numbers: {best_trials_numbers}")

    # mlflow_logger.log_json(best_trials_info, "best_trials_info.json")

    # if len(best_trials_numbers) == 1:
    #     selected_trial = best_trials_numbers[0]
    # else:
    #     # Desempatamos por hidden_size
    #     min_hidden_size = min(best_trials_info, key=lambda x: x['params']['hidden_size'])
    #     selected_trial = min_hidden_size['trial_number']

    # optuna_params = {
    #     "optuna_study_name": study_name,
    #     "objective_metrics": self.metrics,
    #     "best_trials_numbers": best_trials_numbers,
    #     "selected_trial_number": selected_trial,
    #     "metric_best_value": study.trials[selected_trial].value
    # }

    # if "f_beta_score" in self.metrics:
    #     optuna_params["beta"] = self.beta

    # mlflow_logger.log_param("optuna_params", optuna_params)
    # mlflow_logger.log_param("metric_best_value_in_optimization", optuna_params["metric_best_value"])