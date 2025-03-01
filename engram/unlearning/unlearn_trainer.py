import yaml
import argparse
from engram.unlearning import retrain, unlearn


def parse_classes(s):
    """Function to parse a comma-separated string into a list of integers."""
    try:
        return [int(item) for item in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "List of classes should be integers separated by commas"
        )


def get_args():
    parser = argparse.ArgumentParser(description="Unlearning")

    parser.add_argument("--unlearn", type=str, default="retrain")

    # Model settings
    parser.add_argument("--model", type=str, default="cifar_resnet18")
    parser.add_argument("--imagenet_arch", action="store_true")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights for the model.",
    )
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default=None,
        type=str,
    )

    # Optimizer settings
    parser.add_argument("--opt", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--unlearn_lr", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0)

    # Unlearning hyperparameters
    parser.add_argument("--alpha", type=float, default=0.2)

    # Scheduler settings
    parser.add_argument("--sched", type=str, default="cosine")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--unlearn_epochs", type=int, default=5)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--warmup-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--t-in-epochs", action="store_true")

    # Data and general training settings
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument(
        "--class_to_replace",
        type=parse_classes,
        default=[-1],
        help="Specific classes to forget (comma-separated, eg.'1,2,3'. If -1, no forgetting)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="If set, checkpoints will not be saved.",
    )

    # Add the config argument
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (YAML)"
    )
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

    if args.unlearn == "retrain":
        retrain.train(args)
    else:
        unlearn.train(args)
