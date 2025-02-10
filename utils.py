import torch
import timm

from timm import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from . import models
from . import datasets

import os
import random
import numpy as np
import torch

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.num_classes == 10:
        trainloader, testloader = datasets.load_cifar10(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.num_classes == 100:
        trainloader, testloader = datasets.load_cifar100(batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise ValueError("not supported")

    model = create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes).to(device)
    optimizer = create_optimizer_v2(model, opt=args.opt, lr=args.lr)
    scheduler, _ = create_scheduler_v2(optimizer, sched=args.sched)
    criterion = torch.nn.CrossEntropyLoss()

    mixup_fn = timm.data.Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=args.num_classes
    )

    checkpoint_dir = os.path.join(args.output, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(device, model, trainloader, criterion, optimizer, mixup_fn)
        test_loss, test_acc = test_epoch(device, model, testloader, criterion)
        print(f"Epoch [{epoch+1:3}/{args.epochs}] | "
              f"Loss: {train_loss:.4f}/{test_loss:.4f} | "
              f"Acc: {train_acc:5.2f}/{test_acc:5.2f}% ")

        # 매 에포크마다 체크포인트 저장
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'args': vars(args),
        }, checkpoint_path)

        # 현재 에포크의 성능이 최고일 경우 best model 저장
        if test_acc > best_acc:
            best_acc = test_acc
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'args': vars(args),
            }, best_checkpoint_path)

        scheduler.step(epoch)


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
