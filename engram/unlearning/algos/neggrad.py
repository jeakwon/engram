import time
from itertools import cycle
from engram.misc import AverageMeter, warmup_lr, accuracy

from .impl import iterative_unlearn


@iterative_unlearn
def negative_grad(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    # model_ng = freeze_params(model_ng, args)
    f_loader = data_loaders["forget"]
    r_loader = data_loaders["retain"]
    print(f"len(r_loader): {len(r_loader)}, len(f_loader): {len(f_loader)}")

    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    start = time.time()

    for idx, ((input, target), (del_input, del_target)) in enumerate(
        zip(r_loader, cycle(f_loader))
    ):
        if epoch < args.warmup:
            warmup_lr(
                epoch, idx + 1, optimizer, one_epoch_step=len(r_loader), args=args
            )

        # input = input.float()
        # del_input = del_input.float()
        input = input.cuda()
        target = target.cuda()
        del_input = del_input.cuda()
        del_target = del_target.cuda()

        # ===================forward=====================
        output_clean = model(input)
        del_output_clean = model(del_input)
        r_loss = criterion(output_clean, target)
        del_loss = criterion(del_output_clean, del_target)

        loss = args.alpha * r_loss - (1 - args.alpha) * del_loss

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
                    # print(mask[name])

        optimizer.step()

        # ===================meters=====================
        output = output_clean.float()
        loss = loss.float()
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    # model_ng = unfreeze_params(model)

    return top1.avg
