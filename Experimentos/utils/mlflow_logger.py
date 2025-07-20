import mlflow
import tempfile
import json
import os
import time

from pathlib import Path

class MLflowLogger:
    def __init__(
        self, enable_logging, tracking_server_uri, experiment_name, experiment_tags
    ):
        self.enable_logging = enable_logging
        if enable_logging:
            mlflow.set_tracking_uri(tracking_server_uri)
            try:
                mlflow.create_experiment(name=experiment_name, tags=experiment_tags)
            except mlflow.exceptions.RestException as e:
                print(e)
            mlflow.set_experiment(experiment_name)
            self.active_run = self.set_active_run()
        else:
            self.active_run = None

    def start_new_run(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run = mlflow.start_run(run_name=f"run_{timestamp}")
        return run

    def set_active_run(self):
        active_run = mlflow.active_run()
        if not active_run:
            run = self.start_new_run()
        else:
            print(f"There is an active run, name: {active_run.info.run_name}")
            end_run = input("Do you want to end it? (y/n): ")
            if end_run == "y":
                mlflow.end_run()
                run = self.start_new_run()
            else:
                run = active_run
        return run

    def end_run(self):
        if self.active_run:
            mlflow.end_run()

    def log_param(self, key, value):
        if self.enable_logging:
            mlflow.log_param(key, value)

    def log_params(self, params):
        if self.enable_logging:
            mlflow.log_params(params)

    def log_metric(self, key, value):
        if self.enable_logging:
            mlflow.log_metric(key, value)

    def log_artifact(self, path, artifact_path=None):
        if self.enable_logging:
            mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_json(self, json_content, filename, artifact_path=None):
        if self.enable_logging:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, filename)
                with open(file_path, "w") as f:
                    json.dump(json_content, f)
                mlflow.log_artifact(file_path, artifact_path=artifact_path)

    def log_plot(self, fig, filename, artifact_path=None):
        if self.enable_logging:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp, filename)
                # Check the type of the figure and save accordingly
                if hasattr(fig, "savefig"):  # Matplotlib figure
                    fig.savefig(path)
                elif hasattr(fig, "write_image"):  # Plotly figure
                    fig.write_image(path)
                else:
                    raise ValueError(f"Unsupported figure type: {type(fig)}")
                mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_model_architecture(self, model):
        if self.enable_logging:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp, "model_architecture.txt")
                path.write_text(str(model))
                mlflow.log_artifact(path)

    def log_input(self, df, context, target_label='target'):
        if self.enable_logging:
            mlflow.log_input(
                mlflow.data.from_pandas(df, targets=target_label),
                context=context,
            )

    def log_model(self, model, model_name, input_example):
        if self.enable_logging:
            mlflow.pytorch.log_model(
                model.to("cpu"),
                model_name,
                pip_requirements=None,
                input_example=input_example
            )
