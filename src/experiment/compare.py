"""
Experiment comparison and analysis tools.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ExperimentComparator:
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def load_experiment(self, experiment_path: str, name: Optional[str] = None):
        exp_path = Path(experiment_path)
        if not exp_path.exists():
            print(f"Experiment not found: {exp_path}")
            return
        exp_name = name or exp_path.name

        metadata = {}
        config = {}
        metrics = {}
        epoch_metrics = []

        mf = exp_path / "metadata.json"
        if mf.exists():
            metadata = json.load(open(mf))
        cf = exp_path / "config.json"
        if cf.exists():
            config = json.load(open(cf))
        ms = exp_path / "logs" / "metrics_summary.json"
        if ms.exists():
            metrics = json.load(open(ms))
        em = exp_path / "logs" / "epoch_metrics.jsonl"
        if em.exists():
            with open(em) as f:
                for line in f:
                    epoch_metrics.append(json.loads(line))

        self.experiments[exp_name] = {
            "path": str(exp_path),
            "metadata": metadata,
            "config": config,
            "metrics": metrics,
            "epoch_metrics": epoch_metrics,
        }
        print(f"Loaded experiment: {exp_name}")

    def load_all_experiments(self, pattern: str = "*"):
        if not self.experiments_dir.exists():
            print(f"Experiments dir not found: {self.experiments_dir}")
            return
        for d in self.experiments_dir.glob(pattern):
            if d.is_dir():
                self.load_experiment(str(d))
        print(f"Loaded {len(self.experiments)} experiments")

    def get_comparison_table(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        if not self.experiments:
            return pd.DataFrame()
        rows = []
        for name, data in self.experiments.items():
            row = {"experiment": name}
            md = data.get("metadata", {})
            row["status"] = md.get("status", "unknown")
            row["duration_hours"] = md.get("duration_seconds", 0) / 3600
            cfg = data.get("config", {})
            if "training" in cfg:
                row["num_epochs"] = cfg["training"].get("num_epochs", "N/A")
                if "data" in cfg:
                    row["batch_size"] = cfg["data"].get("batch_size", "N/A")
                if "optimizer" in cfg:
                    row["learning_rate"] = cfg["optimizer"].get("learning_rate", "N/A")
            mdict = data.get("metrics", {})
            if metrics is None:
                for m, info in mdict.items():
                    if isinstance(info, dict) and "final" in info:
                        row[f"{m}_final"] = info["final"]
                        row[f"{m}_best"] = info.get("best")
            else:
                for m in metrics:
                    if m in mdict and isinstance(mdict[m], dict):
                        row[f"{m}_final"] = mdict[m].get("final")
                        row[f"{m}_best"] = mdict[m].get("best")
            rows.append(row)
        return pd.DataFrame(rows)

    def print_comparison(self, metrics: Optional[List[str]] = None):
        df = self.get_comparison_table(metrics)
        if df.empty:
            print("No experiments loaded.")
            return
        print("\n" + "=" * 120)
        print("EXPERIMENT COMPARISON")
        print("=" * 120)
        print(df.to_string(index=False))
        print("=" * 120)

    def plot_comparison(self, metric: str = "train_loss", save_path: Optional[str] = None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for name, data in self.experiments.items():
            mdict = data.get("metrics", {})
            if metric in mdict and isinstance(mdict[metric], dict) and "values" in mdict[metric]:
                steps = mdict.get("steps", list(range(len(mdict[metric]["values"]))))
                ax1.plot(steps, mdict[metric]["values"], label=name, alpha=0.8)
        ax1.set_title(f"{metric} over steps")
        ax1.set_xlabel("Step")
        ax1.set_ylabel(metric)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        names, finals, bests = [], [], []
        for name, data in self.experiments.items():
            mdict = data.get("metrics", {})
            if metric in mdict and isinstance(mdict[metric], dict):
                names.append(name)
                finals.append(mdict[metric].get("final", 0))
                bests.append(mdict[metric].get("best", 0))
        if names:
            x = np.arange(len(names))
            w = 0.35
            ax2.bar(x - w / 2, finals, w, label="Final")
            ax2.bar(x + w / 2, bests, w, label="Best")
            ax2.set_xticks(x)
            ax2.set_xticklabels(names, rotation=45, ha="right")
            ax2.set_title(f"{metric}: Final vs Best")
            ax2.legend()
            ax2.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def find_best_experiment(self, metric: str = "val_loss", minimize: bool = True) -> Optional[str]:
        best_name = None
        best_value = float("inf") if minimize else float("-inf")
        for name, data in self.experiments.items():
            mdict = data.get("metrics", {})
            if metric in mdict and isinstance(mdict[metric], dict):
                value = mdict[metric].get("best", mdict[metric].get("final"))
                if value is not None:
                    if (minimize and value < best_value) or (not minimize and value > best_value):
                        best_value = value
                        best_name = name
        if best_name:
            print(f"Best experiment: {best_name} ({metric}={best_value:.4f})")
        return best_name

    def export_comparison(self, output_file: str):
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = self.get_comparison_table()
        if out.suffix == ".csv":
            df.to_csv(out, index=False)
        elif out.suffix == ".json":
            df.to_json(out, orient="records", indent=2)
        else:
            print(f"Unsupported format: {out.suffix}")
            return
        print(f"Exported comparison to {out}")


