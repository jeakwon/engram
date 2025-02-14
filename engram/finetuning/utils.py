import torch
import timm

from timm import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from engram import models
from engram import datasets

import os
import random
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_classes == 10:
        trainloader, testloader = datasets.load_cifar10(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
    elif args.num_classes == 100:
        trainloader, testloader = datasets.load_cifar100(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
    else:
        raise ValueError("not supported")

    model = create_model(
        args.model, pretrained=args.pretrained, num_classes=args.num_classes
    ).to(device)
    optimizer = create_optimizer_v2(
        model, opt=args.opt, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler, _ = create_scheduler_v2(optimizer, sched=args.sched)
    criterion = torch.nn.CrossEntropyLoss()

    mixup_fn = timm.data.Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.1,
        num_classes=args.num_classes,
    )

    log_dir = os.path.join(args.output, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_dir = os.path.join(args.output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            device, model, trainloader, criterion, optimizer, mixup_fn
        )
        test_loss, test_acc = test_epoch(device, model, testloader, criterion)
        print(
            f"Epoch [{epoch+1:3}/{args.epochs}] | Loss: {train_loss:.4f}/{test_loss:.4f} | Acc: {train_acc:5.2f}/{test_acc:5.2f}% "
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Test", test_acc, epoch)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "args": vars(args),
            },
            checkpoint_path,
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state_dict_path = os.path.join(checkpoint_dir, "best_state_dict.pt")
            torch.save(model.state_dict(), best_state_dict_path)

        scheduler.step(epoch)

    writer.close()


def train_epoch(device, model, trainloader, criterion, optimizer, mixup_fn):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Apply MixUp
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
        correct += predicted.eq(labels.argmax(dim=1)).sum().item()

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
