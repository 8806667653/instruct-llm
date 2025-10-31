"""
Real-time training monitoring and visualization tools.
"""

import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


class TrainingMonitor:
    def __init__(
        self,
        experiment_name: str = "training",
        enable_plots: bool = True,
        plot_freq: int = 100,
        early_stopping_patience: Optional[int] = None,
        early_stopping_delta: float = 0.001,
    ):
        self.experiment_name = experiment_name
        self.enable_plots = enable_plots
        self.plot_freq = plot_freq
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.steps: List[int] = []
        self.timestamps: List[float] = []

        self.start_time = time.time()
        self.last_step_time = time.time()

        self.custom_metrics: Dict[str, List[float]] = {}

        self.fig = None
        self.axes = None
        if self.enable_plots:
            plt.ion()
            self._setup_plots()

    def _setup_plots(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle(f"Training Monitor: {self.experiment_name}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    def log_step(self, step: int, train_loss: Optional[float] = None, val_loss: Optional[float] = None, **kwargs):
        current_time = time.time()
        self.steps.append(step)
        self.timestamps.append(current_time - self.start_time)

        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

        for k, v in kwargs.items():
            self.custom_metrics.setdefault(k, []).append(v)

        step_time = current_time - self.last_step_time
        self.last_step_time = current_time

        self._print_progress(step, train_loss, val_loss, step_time, kwargs)

        if self.enable_plots and step % self.plot_freq == 0:
            self._update_plots()

        if val_loss is not None and self.early_stopping_patience is not None:
            return self._check_early_stopping(val_loss)
        return False

    def _print_progress(self, step: int, train_loss, val_loss, step_time: float, custom_metrics: Dict):
        msg = f"Step {step:6d} | "
        if train_loss is not None:
            msg += f"Train Loss: {train_loss:.4f} | "
        if val_loss is not None:
            msg += f"Val Loss: {val_loss:.4f} | "
        msg += f"Speed: {step_time:.3f}s/step"
        if custom_metrics:
            extra = " | ".join(f"{k}: {v:.4f}" for k, v in custom_metrics.items())
            msg += f" | {extra}"
        print(msg)

    def _update_plots(self):
        if self.fig is None:
            return
        for ax in self.axes.flat:
            ax.clear()

        ax1 = self.axes[0, 0]
        if self.train_losses:
            ax1.plot(self.steps[: len(self.train_losses)], self.train_losses, label="Train Loss", color="blue", alpha=0.7)
        if self.val_losses:
            ax1.plot(self.steps[: len(self.val_losses)], self.val_losses, label="Val Loss", color="red", alpha=0.7, linewidth=2)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = self.axes[0, 1]
        if self.train_losses and self.timestamps:
            th = [t / 3600 for t in self.timestamps[: len(self.train_losses)]]
            ax2.plot(th, self.train_losses, label="Train Loss", color="blue", alpha=0.7)
        if self.val_losses:
            vh = [t / 3600 for t in self.timestamps[: len(self.val_losses)]]
            ax2.plot(vh, self.val_losses, label="Val Loss", color="red", alpha=0.7, linewidth=2)
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss vs Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = self.axes[1, 0]
        for metric_name, values in self.custom_metrics.items():
            if "acc" in metric_name.lower():
                ax3.plot(self.steps[: len(values)], values, label=metric_name, alpha=0.7)
        if self.custom_metrics:
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Metric Value")
            ax3.set_title("Additional Metrics")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No additional metrics", ha="center", va="center", transform=ax3.transAxes)

        ax4 = self.axes[1, 1]
        if len(self.steps) > 1:
            diffs = [self.timestamps[i] - self.timestamps[i - 1] for i in range(1, len(self.timestamps))]
            sps = [1.0 / d if d > 0 else 0 for d in diffs]
            ax4.plot(self.steps[1 : len(sps) + 1], sps, color="green", alpha=0.7)
            ax4.set_xlabel("Step")
            ax4.set_ylabel("Steps/Second")
            ax4.set_title("Training Speed")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.draw()
        plt.pause(0.001)

    def _check_early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss - self.early_stopping_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        if self.patience_counter >= self.early_stopping_patience:
            print(f"\nEarly stopping triggered after {self.patience_counter} steps without improvement")
            return True
        return False

    def save_plots(self, save_path: str):
        if self.fig is not None:
            self.fig.savefig(save_path, dpi=300, bbox_inches="tight")

    def get_summary(self) -> Dict:
        duration = time.time() - self.start_time
        summary = {"total_steps": len(self.steps), "duration_seconds": duration, "duration_hours": duration / 3600}
        if self.train_losses:
            summary["final_train_loss"] = self.train_losses[-1]
            summary["best_train_loss"] = min(self.train_losses)
        if self.val_losses:
            summary["final_val_loss"] = self.val_losses[-1]
            summary["best_val_loss"] = min(self.val_losses)
        if len(self.steps) > 1:
            summary["avg_steps_per_second"] = len(self.steps) / duration
        return summary

    def print_summary(self):
        s = self.get_summary()
        print("\n" + "=" * 80)
        print(f"Training Summary: {self.experiment_name}")
        print("=" * 80)
        for k, v in s.items():
            print(f"{k}: {v}")
        print("=" * 80)






