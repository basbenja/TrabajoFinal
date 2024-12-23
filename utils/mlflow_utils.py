import mlflow
import time
import tempfile
import json
import os

from pathlib import Path

def start_new_run():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run = mlflow.start_run(run_name=f"run_{timestamp}")
    return run

def set_active_run():
    active_run = mlflow.active_run()
    if not active_run:
        run = start_new_run()
    else:
        print(f"There is an active run, name: {active_run.info.run_name}")
        end_run = input("Do you want to end it? (y/n): ")
        if end_run == "y":
            mlflow.end_run()
            run = start_new_run()
        else:
            run = active_run
    return run

def log_plot(fig, filename):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, filename)
        # Check the type of the figure and save accordingly
        if hasattr(fig, "savefig"):  # Matplotlib figure
            fig.savefig(path)
        elif hasattr(fig, "write_image"):  # Plotly figure
            fig.write_image(path)
        else:
            raise ValueError(f"Unsupported figure type: {type(fig)}")
        mlflow.log_artifact(path)

def log_json(json_content, filename):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, filename)
        with open(file_path, "w") as f:
            json.dump(json_content, f)
        mlflow.log_artifact(file_path)