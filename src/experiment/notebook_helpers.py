"""
Notebook helpers for easy experimentation in Jupyter notebooks.
"""

from typing import Dict, Any, Optional

import torch

from .config import ExperimentConfig
from .tracker import ExperimentTracker
from .evaluator import LLMEvaluator
from .monitor import TrainingMonitor


class NotebookExperiment:
    def __init__(self, experiment_name: str, config: Optional[ExperimentConfig] = None, config_path: Optional[str] = None):
        self.experiment_name = experiment_name
        self.config = config or (ExperimentConfig.load(config_path) if config_path else ExperimentConfig(experiment_name=experiment_name))
        self.tracker: Optional[ExperimentTracker] = None
        self.monitor: Optional[TrainingMonitor] = None
        self.evaluator: Optional[LLMEvaluator] = None

    def setup_tracking(self):
        self.tracker = ExperimentTracker(self.experiment_name, output_dir=self.config.output_dir, config=self.config)
        return self.tracker

    def setup_monitoring(self, enable_plots: bool = True):
        self.monitor = TrainingMonitor(experiment_name=self.experiment_name, enable_plots=enable_plots, plot_freq=self.config.training.eval_freq)
        return self.monitor

    def setup_evaluator(self, model, tokenizer, device: Optional[str] = None):
        device = device or self.config.training.device
        self.evaluator = LLMEvaluator(model, tokenizer, device)
        return self.evaluator

    def quick_setup(self, model, tokenizer, device: Optional[str] = None):
        self.setup_tracking()
        self.setup_monitoring()
        self.setup_evaluator(model, tokenizer, device)
        return self.tracker, self.monitor, self.evaluator

    def train_step(self, model, batch, optimizer, step: int, device: Optional[str] = None) -> Dict[str, float]:
        device = device or self.config.training.device
        model.train()
        optimizer.zero_grad()
        if len(batch) == 2:
            input_batch, target_batch = batch
            attention_mask = None
        else:
            input_batch, target_batch, attention_mask = batch
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            logits = model(input_batch, attention_mask=attention_mask)
        else:
            logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target_batch.reshape(-1), ignore_index=-100)
        loss.backward()
        if self.config.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.max_grad_norm)
        optimizer.step()
        return {"train_loss": loss.item()}

    def eval_step(self, model, val_loader, step: int) -> Dict[str, float]:
        if self.evaluator is None:
            raise RuntimeError("Evaluator not initialized.")
        return self.evaluator.evaluate_loader(val_loader, num_batches=self.config.training.eval_iter, verbose=False)

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if self.tracker:
            self.tracker.log_metrics(metrics, step)
        if self.monitor:
            train_loss = metrics.get("train_loss")
            val_loss = metrics.get("val_loss") or metrics.get("loss")
            other = {k: v for k, v in metrics.items() if k not in ["train_loss", "val_loss", "loss"]}
            self.monitor.log_step(step, train_loss, val_loss, **other)

    def save_checkpoint(self, model, optimizer, step: int, metrics: Optional[Dict[str, float]] = None):
        if not self.tracker:
            return
        is_best = False
        if metrics and "val_loss" in metrics:
            is_best = self.tracker.should_save_as_best(metrics["val_loss"])
        self.tracker.save_checkpoint(model, optimizer, step, metrics, is_best)

    def finalize(self, status: str = "completed"):
        if self.tracker:
            self.tracker.finalize(status)
        if self.monitor:
            self.monitor.print_summary()






