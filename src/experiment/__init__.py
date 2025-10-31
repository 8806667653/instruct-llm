"""
Experiment tracking and evaluation framework for LLM training.

This module provides a plug-and-play system for:
- Configuration management
- Experiment tracking
- Multi-metric evaluation
- Checkpoint management
- Visualization and comparison
"""

from .config import ExperimentConfig, load_config, save_config
from .tracker import ExperimentTracker
from .evaluator import LLMEvaluator
from .monitor import TrainingMonitor
from .compare import ExperimentComparator

__all__ = [
    "ExperimentConfig",
    "load_config",
    "save_config",
    "ExperimentTracker",
    "LLMEvaluator",
    "TrainingMonitor",
    "ExperimentComparator",
]






