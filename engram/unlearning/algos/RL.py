import time
from copy import deepcopy

import numpy as np
import torch
from engram.misc import AverageMeter, warmup_lr, accuracy

from .impl import iterative_unlearn


@iterative_unlearn
def RL(data_loaders, model, criterion, optimizer, epoch, args, mask=None):

    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    if args.dataset == "cifar10":
        forget_dataset.targets = np.random.randint(
            0, args.num_classes, forget_dataset.targets.shape
        )
    elif args.dataset == "svhn":
        forget_dataset.labels = np.random.randint(
            0, args.num_classes, forget_dataset.labels.shape
        )

    retain_dataset = retain_loader.dataset
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    loader_len = len(forget_loader) + len(retain_loader)

    if epoch < args.warmup:
        warmup_lr(epoch, i + 1, optimizer, one_epoch_step=loader_len, args=args)

    for it, (image, target) in enumerate(train_loader):
        i = it + len(forget_loader)
        image = image.cuda()
        target = target.cuda()
        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            # print("Applying mask for SalUn!")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
                    # print(mask[name])
        else:
            # print("No mask applied!")
            pass

        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

    return top1.avg
