import timm
import torch

from engram.datasets import load_cifar10, load_cifar100

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training with TIMM")

    # Model settings
    parser.add_argument('--model', type=str, default='cifar_resnet18', 
                        help='Model architecture to use (e.g., cifar_resnet18, cifar_vit_base_patch16_224, cifar_mixer_b16_224)')
    parser.add_argument('--pretrained', action='store_true', 
                        help='Load pretrained weights')
    parser.add_argument('--num-classes', type=int, default=10, 
                        help='Number of classes in the dataset')

    # Optimizer settings
    parser.add_argument('--opt', type=str, default='lion', 
                        help='Optimizer to use (e.g., adamw, lion)')
    parser.add_argument('--lr', type=float, default=1e-5, 
                        help='Base learning rate')

    # Scheduler settings
    parser.add_argument('--sched', type=str, default='cosine', 
                        help='Learning rate scheduler type (e.g., cosine)')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Total number of training epochs')
    parser.add_argument('--lr-min', type=float, default=1e-5, 
                        help='Minimum learning rate')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, 
                        help='Warmup learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=30, 
                        help='Number of warmup epochs')
    parser.add_argument('--t-in-epochs', action='store_true', 
                        help='Schedule time unit is in epochs (if not, then in steps)')

    # Data and general training settings
    parser.add_argument('--batch-size', type=int, default=100, 
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=2, 
                        help='Batch size for training')
    parser.add_argument('--data-path', type=str, default='./data', 
                        help='Path to the dataset')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--output', type=str, default='./output', 
                        help='Output directory for checkpoints and logs')

    try:
        get_ipython()  # Check for IPython environment
        args = parser.parse_args([]) # Pass an empty list when in a notebook
    except NameError:
        args = parser.parse_args()  # Parse command-line arguments otherwise

    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_classes == 10:
        trianloader, testloader = load_cifar10(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.num_classes == 100:
        trianloader, testloader = load_cifar100(batch_size=args.batch_size, num_workers=args.num_workers)

    model = model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes).to(device)
    optimizer = timm.optim.create_optimizer_v2(model, opt=args.opt, lr=args.lr)
    scheduler, _ = timm.scheduler.create_scheduler_v2(optimizer, sched=args.sched)
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

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(device, model, trianloader, criterion, optimizer, mixup_fn)
        test_loss, test_acc = test_epoch(device, model, testloader, criterion)
        print(f"Epoch [{epoch+1:3}/{args.num_epochs}] | "
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
