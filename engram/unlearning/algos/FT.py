import sys
import time

import torch

from .impl import iterative_unlearn
from engram.misc import AverageMeter, warmup_lr, accuracy

sys.path.append(".")
try:
    from imagenet import get_x_y_from_data_dict
except ImportError:
    pass


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def FT_iter(
    data_loaders, model, criterion, optimizer, epoch, args, mask=None, with_l1=False
):

    train_loader = data_loaders["retain"]

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            if with_l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
                        # print(mask[name])

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            if with_l1:
                loss += args.alpha * l1_regularization(model)

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
                        # print(mask[name])

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


@iterative_unlearn
def FT(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    return FT_iter(data_loaders, model, criterion, optimizer, epoch, args, mask)


@iterative_unlearn
def FT_l1(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    return FT_iter(
        data_loaders, model, criterion, optimizer, epoch, args, mask, with_l1=True
    )


# Fine-Tuning with L1 Regularization
# with_l1=True: add a penalty to the loss function based on absolute values of the model parameters (weights)
# L1 regularization encourages the model to have a sparse representation
