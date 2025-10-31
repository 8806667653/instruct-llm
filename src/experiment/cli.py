"""
Command-line interface for experiment management.
"""

import argparse
from pathlib import Path

from .config import ExperimentConfig, create_default_configs
from .compare import ExperimentComparator


def create_config_command(args):
    config = ExperimentConfig(experiment_name=args.name, description=args.description or "")
    if args.learning_rate:
        config.optimizer.learning_rate = args.learning_rate
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    output_path = args.output or f"configs/{args.name}.yaml"
    config.save(output_path, format=args.format)
    print(f"Created config: {output_path}")


def list_experiments_command(args):
    d = Path(args.dir)
    if not d.exists():
        print(f"Experiments dir not found: {d}")
        return
    exps = [p for p in sorted(d.glob("*")) if p.is_dir()]
    print(f"\nFound {len(exps)} experiments in {d}")
    print("=" * 80)
    for e in exps:
        print(e.name)
    print("=" * 80)


def compare_experiments_command(args):
    comparator = ExperimentComparator(experiments_dir=args.dir)
    if args.experiments:
        for exp in args.experiments:
            comparator.load_experiment(str(Path(args.dir) / exp))
    else:
        comparator.load_all_experiments()
    metrics = args.metrics.split(",") if args.metrics else None
    comparator.print_comparison(metrics=metrics)
    if args.find_best:
        comparator.find_best_experiment(metric=args.find_best)
    if args.export:
        comparator.export_comparison(args.export)


def show_experiment_command(args):
    comparator = ExperimentComparator(experiments_dir=args.dir)
    comparator.load_experiment(str(Path(args.dir) / args.experiment))
    comparator.print_comparison()


def create_templates_command(args):
    create_default_configs()


def main():
    parser = argparse.ArgumentParser(description="LLM Experiment Management CLI")
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("create-config", help="Create new experiment configuration")
    p.add_argument("--name", required=True)
    p.add_argument("--description")
    p.add_argument("--output")
    p.add_argument("--format", choices=["yaml", "json"], default="yaml")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--epochs", type=int)
    p.set_defaults(func=create_config_command)

    p = subparsers.add_parser("list", help="List all experiments")
    p.add_argument("--dir", default="experiments")
    p.set_defaults(func=list_experiments_command)

    p = subparsers.add_parser("compare", help="Compare experiments")
    p.add_argument("--dir", default="experiments")
    p.add_argument("--experiments", nargs="+")
    p.add_argument("--metrics")
    p.add_argument("--find-best")
    p.add_argument("--export")
    p.set_defaults(func=compare_experiments_command)

    p = subparsers.add_parser("show", help="Show experiment details")
    p.add_argument("--dir", default="experiments")
    p.add_argument("--experiment", required=True)
    p.set_defaults(func=show_experiment_command)

    subparsers.add_parser("create-templates", help="Create default templates").set_defaults(func=create_templates_command)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()






