"""
Module `ops.mlops.tracking` - ExperimentTracker (đơn giản, file-based experiment tracking).

Important keywords: Args, Methods, Returns, Notes
"""
import os
import sys
import platform
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from ...utils import IOHandler, ensure_dir


class ExperimentTracker:
    """
    Simple experiment tracker thay thế cho MLflow (file-based).

    Methods:
        start_run(run_name), end_run(status), log_params, log_metrics, log_artifact, get_run_info
    """

    def __init__(self, base_dir: str = "artifacts/experiments"):
        self.base_dir = base_dir
        self.experiments_file = os.path.join(base_dir, "experiments.csv")
        ensure_dir(base_dir)

        if not os.path.exists(self.experiments_file):
            # Create empty dataframe with object dtype for string columns to avoid dtype coercion later
            cols = ["run_id", "run_name", "start_time", "end_time", "status", "duration_seconds"]
            empty = {c: pd.Series(dtype='object') for c in cols}
            pd.DataFrame(empty).to_csv(self.experiments_file, index=False)

        self.current_run_id = None
        self.current_run_dir = None
        self.run_start_time = None

    def start_run(self, run_name: str = None) -> str:
        """Start a new run."""
        self.run_start_time = datetime.now()
        if run_name is None:
            run_name = self.run_start_time.strftime("%Y%m%d_%H%M%S")
        self.current_run_id = run_name

        self.current_run_dir = os.path.join(self.base_dir, self.current_run_id)
        ensure_dir(self.current_run_dir)

        new_row = pd.DataFrame([{
            "run_id": self.current_run_id,
            "run_name": run_name,
            "start_time": self.run_start_time.isoformat(),
            "end_time": None,
            "status": "RUNNING",
            "duration_seconds": None
        }])

        try:
            experiments_df = pd.read_csv(self.experiments_file, dtype=str)
            # Align new_row columns with experiments_df to avoid dtype issues
            new_row = new_row.reindex(columns=experiments_df.columns)
            # Append by direct assignment (avoid pd.concat warnings)
            row_dict = new_row.iloc[0].to_dict()
            experiments_df.loc[len(experiments_df)] = row_dict
            experiments_df.to_csv(self.experiments_file, index=False)
        except Exception as e:
            print(f"Error logging run start: {e}")

        return self.current_run_id

    def end_run(self, status: str = "FINISHED"):
        """End current run."""
        if self.current_run_id is None:
            return

        end_time = datetime.now()
        duration = (end_time - self.run_start_time).total_seconds()

        try:
            experiments_df = pd.read_csv(self.experiments_file, dtype=str)
            # Ensure time columns are treated as object (strings) to avoid setting incompatible dtypes
            for tcol in ["start_time", "end_time"]:
                if tcol in experiments_df.columns:
                    experiments_df[tcol] = experiments_df[tcol].astype(object)
            mask = experiments_df["run_id"] == self.current_run_id
            experiments_df.loc[mask, "end_time"] = end_time.isoformat()
            experiments_df.loc[mask, "status"] = status
            experiments_df.loc[mask, "duration_seconds"] = duration
            experiments_df.to_csv(self.experiments_file, index=False)
        except Exception as e:
            print(f"Error logging run end: {e}")

        self.current_run_id = None
        self.current_run_dir = None
        self.run_start_time = None

    def log_params(self, params: Dict[str, Any]):
        """Save parameters."""
        if self.current_run_dir is None:
            return

        params_file = os.path.join(self.current_run_dir, "params.json")
        IOHandler.save_json(params, params_file)

    def log_metrics(self, metrics: Dict[str, float]):
        """Save metrics."""
        if self.current_run_dir is None:
            return

        metrics_file = os.path.join(self.current_run_dir, "metrics.json")

        if os.path.exists(metrics_file):
            existing = IOHandler.load_json(metrics_file)
            existing.update(metrics)
            metrics = existing

        IOHandler.save_json(metrics, metrics_file)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Copy artifact to run directory."""
        if self.current_run_dir is None or not os.path.exists(local_path):
            return

        import shutil
        artifacts_dir = os.path.join(self.current_run_dir, "artifacts")
        ensure_dir(artifacts_dir)

        try:
            if os.path.isfile(local_path):
                dest = os.path.join(artifacts_dir, os.path.basename(local_path))
                shutil.copy2(local_path, dest)
            elif os.path.isdir(local_path):
                dest = os.path.join(artifacts_dir, os.path.basename(local_path))
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.copytree(local_path, dest)
        except Exception as e:
            print(f"Error logging artifact {local_path}: {e}")

    def get_run_info(self, run_id: str = None) -> Dict:
        """Get run info."""
        if run_id is None:
            run_id = self.current_run_id

        if run_id is None:
            return {}

        run_dir = os.path.join(self.base_dir, run_id)

        info = {
            "run_id": run_id,
            "run_dir": run_dir
        }

        params_file = os.path.join(run_dir, "params.json")
        if os.path.exists(params_file):
            info["params"] = IOHandler.load_json(params_file)

        metrics_file = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_file):
            info["metrics"] = IOHandler.load_json(metrics_file)

        return info

    def log_metadata(self, metadata: Dict[str, Any]):
        """Log environment info."""
        if self.current_run_dir is None:
            return

        auto_metadata = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp": datetime.now().isoformat(),
        }

        full_metadata = {**auto_metadata, **metadata}

        metadata_file = os.path.join(self.current_run_dir, "metadata.json")
        IOHandler.save_json(full_metadata, metadata_file)
