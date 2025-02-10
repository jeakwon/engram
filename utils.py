import torch
import timm

from timm import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from . import models
from . import datasets


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_classes == 10:
        trianloader, testloader = datasets.load_cifar10(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.num_classes == 100:
        trianloader, testloader = datasets.load_cifar100(batch_size=args.batch_size, num_workers=args.num_workers)

    model = model = create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes).to(device)
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

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(device, model, trianloader, criterion, optimizer, mixup_fn)
        test_loss, test_acc = test_epoch(device, model, testloader, criterion)
        print(f"Epoch [{epoch+1:3}/{args.epochs}] | "
              f"Loss: {train_loss:.4f}/{test_loss:.4f} | "
              f"Acc: {train_acc:5.2f}/{test_acc:5.2f}% ")
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
