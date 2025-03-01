import yaml
import argparse
from engram.finetuning.utils import train


def get_args():
    parser = argparse.ArgumentParser(description="Training with TIMM")

    # Model settings
    parser.add_argument("--model", type=str, default="cifar_resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-classes", type=int, default=10)

    # Optimizer settings
    parser.add_argument("--opt", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)

    # Scheduler settings
    parser.add_argument("--sched", type=str, default="cosine")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--warmup-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--t-in-epochs", action="store_true")

    # Data and general training settings
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./output")

    # Add the config argument
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (YAML)"
    )
    # If running in a notebook, parse empty args.
    try:
        get_ipython()
        args = parser.parse_args([])
    except NameError:
        args = parser.parse_args()

    return args


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def merge_config_args(args, config):
    for key, value in config.items():
        setattr(args, key, value)
    return args


if __name__ == "__main__":
    args = get_args()
    if args.config:
        config = load_config(args.config)
        args = merge_config_args(args, config)

    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")

    train(args)
