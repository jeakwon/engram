import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import copy


def load_cifar10(batch_size=100, seed=1, class_to_replace=None, data_dir="./data"):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
            ),
        ]
    )

    train_set = CIFAR10(data_dir, train=True, transform=transform_train, download=False)
    print("Total train set size:", len(train_set))
    test_set = CIFAR10(data_dir, train=False, transform=transform_test, download=False)
    print("Total test set size:", len(test_set))

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []

    for i in range(10):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )

    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]
    valid_set.transform = transform_test
    print("Validation set size:", len(valid_set))

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]
    print("Train set size:", len(train_set))

    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
        )

    loader_args = {"num_workers": 2, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed + worker_id))
        random.seed(int(seed + worker_id))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def load_cifar100(batch_size=100, seed=1, class_to_replace=None, data_dir="./data"):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    train_set = CIFAR100(
        data_dir, train=True, transform=transform_train, download=False
    )
    print("Total train set size:", len(train_set))
    test_set = CIFAR100(data_dir, train=False, transform=transform_test, download=False)
    print("Total test set size:", len(test_set))

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []

    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )

    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]
    valid_set.transform = transform_test
    print("Validation set size:", len(valid_set))

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]
    print("Train set size:", len(train_set))

    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
        )

    loader_args = {"num_workers": 2, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed + worker_id))
        random.seed(int(seed + worker_id))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def replace_class(
    dataset: torch.utils.data.Dataset,
    class_to_replace: int,
):
    try:
        # indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
        indexes = np.flatnonzero(np.isin(np.array(dataset.targets), class_to_replace))
    except:
        try:
            indexes = np.flatnonzero(np.array(dataset.labels) == class_to_replace)
        except:
            indexes = np.flatnonzero(np.array(dataset._labels) == class_to_replace)

    # Notice the -1 to make class 0 work
    try:
        dataset.targets[indexes] = -dataset.targets[indexes] - 1
    except:
        try:
            dataset.labels[indexes] = -dataset.labels[indexes] - 1
        except:
            dataset._labels[indexes] = -dataset._labels[indexes] - 1
