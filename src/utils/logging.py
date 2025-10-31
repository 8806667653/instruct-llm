"""Logging utilities and experiment tracking."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logger(
    name: str = "instruct-lite",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentTracker:
    """Simple experiment tracker (can be extended with W&B, MLFlow, etc.)."""
    
    def __init__(self, experiment_name: str, log_dir: str = "./training_outputs/logs"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of experiment
            log_dir: Directory for logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = []
        self.logger = setup_logger(
            name=experiment_name,
            log_file=str(self.log_dir / f"{experiment_name}.log")
        )
    
    def log_metric(self, name: str, value: float, step: int):
        """Log a metric."""
        self.metrics.append({
            "name": name,
            "value": value,
            "step": step
        })
        self.logger.info(f"Step {step} - {name}: {value:.4f}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        self.logger.info("=" * 50)
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)
    
    def finish(self):
        """Finish experiment."""
        self.logger.info("Experiment completed")


def init_wandb(project_name: str, config: Dict[str, Any], run_name: Optional[str] = None):
    """
    Initialize Weights & Biases tracking.
    
    Args:
        project_name: W&B project name
        config: Configuration dict
        run_name: Optional run name
    """
    try:
        import wandb
        
        wandb.init(
            project=project_name,
            config=config,
            name=run_name
        )
        print("W&B initialized successfully")
    except ImportError:
        print("W&B not installed. Install with: pip install wandb")


def log_to_wandb(metrics: Dict[str, float], step: int):
    """Log metrics to W&B."""
    try:
        import wandb
        wandb.log(metrics, step=step)
    except ImportError:
        pass

