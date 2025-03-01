import torch
import timm

from timm import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from engram import models
from engram import datasets
from engram.misc import set_seed, save_checkpoint

import os
import torch
import wandb


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    model, data_config = create_model(
        args.model, pretrained=args.pretrained, num_classes=args.num_classes
    )
    model.to(device)
    set_seed(args.seed)
    if args.dataset == "cifar10":
        trainloader, valloader, testloader = datasets.load_cifar10(
            data_config,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    elif args.dataset == "cifar100":
        trainloader, valloader, testloader = datasets.load_cifar100(
            data_config,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        raise ValueError("not supported")

    if args.wandb:
        logger = wandb.init(
            project="engram",
            config=vars(args),
            group=f"finetuning_{args.dataset}_{args.model}_{args.opt}",
        )

    optimizer = create_optimizer_v2(
        model, opt=args.opt, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched=args.sched,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.lr_min,
        warmup_lr=args.warmup_lr,
    )
    criterion = torch.nn.CrossEntropyLoss()

    if args.mixup:
        mixup_fn = timm.data.Mixup(
            mixup_alpha=0.2,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=args.num_classes,
        )
    else:
        mixup_fn = None

    os.makedirs(args.output, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            args, device, model, trainloader, criterion, optimizer, mixup_fn
        )
        val_loss, val_acc = test_epoch(device, model, valloader, criterion)
        print(
            f"Epoch [{epoch+1:3}/{args.epochs}] | Loss: {train_loss:.4f}/{val_loss:.4f} | Acc: {train_acc:5.2f}/{val_acc:5.2f}% "
        )

        test_loss, test_acc = test_epoch(device, model, testloader, criterion)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:5.2f}%")

        if args.wandb:
            logger.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "args": vars(args),
        }
        is_best = val_acc > best_acc
        if val_acc > best_acc:
            best_acc = val_acc
        save_checkpoint(state, is_best, args.output)
        scheduler.step(epoch)

    if args.wandb:
        logger.finish()


def train_epoch(args, device, model, trainloader, criterion, optimizer, mixup_fn):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Apply MixUp
        if args.mixup:
            inputs, labels = mixup_fn(inputs, labels)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track Loss & Accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        if args.mixup:
            correct += predicted.eq(labels.argmax(dim=1)).sum().item()
        else:
            correct += predicted.eq(labels).sum().item()

    running_loss /= len(trainloader)
    accuracy = 100 * correct / total
    return running_loss, accuracy


@torch.no_grad()
def test_epoch(device, model, testloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Track Loss & Accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    running_loss /= len(testloader)
    accuracy = 100 * correct / total
    return running_loss, accuracy
