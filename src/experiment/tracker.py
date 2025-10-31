"""
Experiment tracking with metrics logging, checkpointing, and metadata management.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import torch


class ExperimentTracker:
    """Tracks experiments with automatic logging and checkpointing."""

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "experiments",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.logs_dir = self.exp_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.samples_dir = self.exp_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        self.metrics_history: Dict[str, List[float]] = {}
        self.step_history: List[int] = []
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.current_epoch = 0
        self.start_time = time.time()

        self.config = config
        if config is not None:
            self._save_config()

        self.metadata = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "pytorch_version": torch.__version__,
        }
        self._save_metadata()

    def log_metrics(self, metrics: Dict[str, float], step: int):
        self.global_step = step
        if step not in self.step_history:
            self.step_history.append(step)

        for name, value in metrics.items():
            self.metrics_history.setdefault(name, []).append(value)

        log_entry = {"step": step, "timestamp": time.time() - self.start_time, **metrics}
        with open(self.logs_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        self.current_epoch = epoch
        entry = {"epoch": epoch, "timestamp": time.time() - self.start_time, **metrics}
        with open(self.logs_dir / "epoch_metrics.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": self.current_epoch,
            "metrics": metrics or {},
            "config": self.config,
        }

        torch.save(checkpoint, self.checkpoints_dir / f"checkpoint_step_{step}.pt")
        torch.save(checkpoint, self.checkpoints_dir / "checkpoint_latest.pt")

        if is_best:
            torch.save(checkpoint, self.checkpoints_dir / "checkpoint_best.pt")

        self._cleanup_checkpoints(keep_last=3)

    def should_save_as_best(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False

    def save_generated_samples(self, samples: List[str], step: int):
        with open(self.samples_dir / f"samples_step_{step}.txt", "w") as f:
            f.write("\n".join(samples))

    def finalize(self, status: str = "completed"):
        self.metadata.update(
            {
                "status": status,
                "end_time": datetime.now().isoformat(),
                "duration_seconds": time.time() - self.start_time,
                "total_steps": self.global_step,
                "total_epochs": self.current_epoch,
                "best_val_loss": self.best_val_loss,
            }
        )
        self._save_metadata()
        self._save_metrics_summary()

    def _save_config(self):
        if hasattr(self.config, "to_dict"):
            config_dict = self.config.to_dict()
        else:
            config_dict = self.config
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def _save_metadata(self):
        with open(self.exp_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _save_metrics_summary(self):
        summary = {"steps": self.step_history}
        for m, values in self.metrics_history.items():
            if values:
                summary[m] = {
                    "values": values,
                    "final": values[-1],
                    "best": min(values) if "loss" in m else max(values),
                    "mean": sum(values) / len(values),
                }
        with open(self.logs_dir / "metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def _cleanup_checkpoints(self, keep_last: int = 3):
        checkpoints = sorted(self.checkpoints_dir.glob("checkpoint_step_*.pt"), key=lambda p: p.stat().st_mtime)
        for ckpt in checkpoints[:-keep_last]:
            ckpt.unlink(missing_ok=True)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config"),
    }






