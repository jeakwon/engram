import torch
import numpy as np
import random
import os
import shutil
import torchvision.transforms as transforms
import torch.nn.functional as F
import json
import copy
from torch.utils.data import DataLoader


def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True):
    def _init_fn(worker_id):
        np.random.seed(int(seed + worker_id))
        random.seed(int(seed + worker_id))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=False,
        shuffle=shuffle,
        worker_init_fn=_init_fn if seed is not None else None,
    )


def split_dataset_loader(marked_loader, train_loader_full, args):
    """
    Splits a dataset into two loaders: forget_loader and retain_loader.

    - Forget loader contains samples with negative targets.
    - Retain loader contains samples with non-negative targets.

    Args:
        marked_loader: DataLoader with marked dataset.
        train_loader_full: Full training DataLoader (used for validation check).
        args: Arguments containing seed for reproducibility.

    Returns:
        forget_loader, retain_loader
    """
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    retain_dataset = copy.deepcopy(marked_loader.dataset)

    try:
        marked_forget = forget_dataset.targets < 0
        forget_dataset.data = forget_dataset.data[marked_forget]
        forget_dataset.targets = -forget_dataset.targets[marked_forget] - 1
        forget_loader = replace_loader_dataset(
            forget_dataset, args.batch_size, seed=args.seed, shuffle=True
        )
        print("Forget dataset size:", len(forget_dataset))

        marked_retain = retain_dataset.targets >= 0
        retain_dataset.data = retain_dataset.data[marked_retain]
        retain_dataset.targets = retain_dataset.targets[marked_retain]
        retain_loader = replace_loader_dataset(
            retain_dataset, args.batch_size, seed=args.seed, shuffle=True
        )
        print("Retain dataset size:", len(retain_dataset))

    except AttributeError:
        # Handle cases where dataset uses 'imgs' instead of 'data'
        marked_forget = forget_dataset.targets < 0
        forget_dataset.imgs = [
            img for img, mark in zip(forget_dataset.imgs, marked_forget) if mark
        ]
        forget_dataset.targets = [
            -t - 1 for t, mark in zip(forget_dataset.targets, marked_forget) if mark
        ]
        forget_loader = replace_loader_dataset(
            forget_dataset, args.batch_size, seed=args.seed, shuffle=True
        )
        print("Forget dataset size:", len(forget_dataset))

        marked_retain = retain_dataset.targets >= 0
        retain_dataset.imgs = [
            img for img, mark in zip(retain_dataset.imgs, marked_retain) if mark
        ]
        retain_dataset.targets = [
            t for t, mark in zip(retain_dataset.targets, marked_retain) if mark
        ]
        retain_loader = replace_loader_dataset(
            retain_dataset, args.batch_size, seed=args.seed, shuffle=True
        )
        print("Retain dataset size:", len(retain_dataset))

    assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)

    return forget_loader, retain_loader


def set_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, is_SA_best, save_path, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, "best_state_dict.pth.tar"))


def load_checkpoint(device, save_path, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


def save_unlearn_checkpoint(model, args):
    state = {"state_dict": model.state_dict()}
    filename = "{}_{}_{}_unlearn{}_seed{}.pth.tar".format(
        args.dataset,
        args.model,
        args.opt,
        args.class_to_replace,
        args.seed,
    )
    save_checkpoint(state, False, args.save_dir, filename=filename)
    print("save checkpoint: ", filename)


def save_evals(evaluation_result, args):
    # Convert the evaluation result to JSON string first for proper DataFrame creation
    json_data = json.dumps(evaluation_result)
    filename = "{}_{}_{}_unlearn{}_seed{}.json".format(
        args.dataset,
        args.model,
        args.opt,
        args.class_to_replace,
        args.seed,
    )
    with open(os.path.join(args.save_dir, filename), "w") as f:
        f.write(json_data)


def dataset_convert_to_test(dataset, testset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    while hasattr(testset, "dataset"):
        testset = testset.dataset
    dataset.transform = testset.transform
    dataset.train = False


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):
    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


"""for SCRUB: imported from https://github.com/HobbitLong/RepDistiller"""


class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
