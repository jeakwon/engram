import time
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset

import torch
import timm
import copy

import engram.unlearning.algos as algos

from timm import create_model

from engram import models
from engram import datasets
from engram.misc import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    dataset_convert_to_test,
    split_dataset_loader,
    save_unlearn_checkpoint,
    save_evals,
)
from engram.unlearning.evaluation import SVC_MIA

import os
import torch
import wandb


def train(args):
    start_rte = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if args.dataset == "cifar10":
        train_loader_full, val_loader, _ = datasets.load_cifar10(
            batch_size=args.batch_size,
            seed=args.seed,
        )
        marked_loader, _, test_loader = datasets.load_cifar10(
            batch_size=args.batch_size,
            seed=args.seed,
            class_to_replace=args.class_to_replace,
        )
        args.num_classes = 10
    elif args.dataset == "cifar100":
        train_loader_full, val_loader, _ = datasets.load_cifar100(
            batch_size=args.batch_size,
            seed=args.seed,
        )
        marked_loader, _, test_loader = datasets.load_cifar100(
            batch_size=args.batch_size,
            seed=args.seed,
            class_to_replace=args.class_to_replace,
        )
        args.num_classes = 100
    else:
        raise ValueError("not supported")

    if args.wandb:
        logger = wandb.init(
            project="engram",
            config=vars(args),
            group=f"Unlearn_{args.dataset}_{args.model}_{args.opt}_{args.unlearn}",
        )

    forget_loader, retain_loader = split_dataset_loader(
        marked_loader, train_loader_full, args
    )

    # unlearn_data_loaders 구성: retain, forget, val, test
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = torch.nn.CrossEntropyLoss()

    model = create_model(args.model, pretrained=False, num_classes=args.num_classes).to(
        device
    )
    checkpoint = load_checkpoint(device, args.output)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = torch.nn.CrossEntropyLoss()

    parts = args.output.split("/")
    idx = parts.index("finetuning")
    parts[idx] = "unlearn"
    args.save_dir = os.path.join("/".join(parts), args.unlearn)
    os.makedirs(args.save_dir, exist_ok=True)

    start_unlearn = time.time()
    unlearn_method = algos.get_unlearn_method(args.unlearn)
    if args.unlearn == "SCRUB":
        model_s = copy.deepcopy(model)
        model_t = copy.deepcopy(model)
        module_list = torch.nn.ModuleList([model_s, model_t])
        unlearn_method(unlearn_data_loaders, module_list, criterion, args)
        model = module_list[0]
    else:
        unlearn_method(unlearn_data_loaders, model, criterion, args)

    end_rte = time.time()
    print(f"Overall time (unlearning & preparation): {end_rte - start_rte:.3f}s")
    print(f"Unlearning time: {end_rte - start_unlearn:.3f}s")
    logger.log({"unlearn_time": end_rte - start_unlearn})
    logger.log({"overall_time (unlearning & preparation)": end_rte - start_rte})
    if not args.no_save:
        save_unlearn_checkpoint(model, args)

    # Accuracy 평가
    print("-------------------Start accuracy evaluation-------------------")
    evaluation_result = {}
    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            print("Evaluating on:", name)
            dataset_convert_to_test(loader.dataset, test_loader.dataset)
            _, val_acc = test_epoch(device, model, loader, criterion)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")
        logger.log(
            {
                "forget acc": accuracy["forget"],
                "retain acc": accuracy["retain"],
                "val acc": accuracy["val"],
                "test acc": accuracy["test"],
            }
        )
        evaluation_result["accuracy"] = accuracy

    if not args.no_save:
        save_evals(evaluation_result, args)

    # MIA 평가 (forget efficacy)
    print("-------------------Start MIA evaluation-------------------")
    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        evaluation_result.pop(deprecated, None)
    MIA_forget_efficacy = True
    if MIA_forget_efficacy and "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        dataset_convert_to_test(retain_loader, test_loader.dataset)
        dataset_convert_to_test(forget_loader, test_loader.dataset)
        dataset_convert_to_test(test_loader, test_loader.dataset)
        shadow_train = Subset(retain_loader.dataset, list(range(test_len)))
        shadow_train_loader = DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
        evaluation_result["SVC_MIA_forget_efficacy"] = SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        logger.log(
            {"SVC_MIA_forget_efficacy": evaluation_result["SVC_MIA_forget_efficacy"]}
        )

    if not args.no_save:
        save_evals(evaluation_result, args)

    print("-------------------Start Linear Probing (LP) evaluation-------------------")
    # Freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False
    model.head = torch.nn.Linear(model.head.in_features, args.num_classes).to(device)

    # Train linear classifier with simple Adam optimizer
    lp_optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-4)

    # Train for 3 epochs
    print("Train")
    for epoch in range(3):
        _, _ = train_epoch(
            args, device, model, train_loader_full, criterion, lp_optimizer
        )

    # Evaluate on training sets
    print("Eval")
    _, retain_acc = test_epoch(device, model, retain_loader, criterion)
    _, forget_acc = test_epoch(device, model, forget_loader, criterion)

    # Evaluate on test set's retain and forget classes
    targets = (
        test_loader.dataset.targets
        if hasattr(test_loader.dataset, "targets")
        else test_loader.dataset.tensors[1]
    )
    retain_indices = [
        i for i, label in enumerate(targets) if label not in args.class_to_replace
    ]
    forget_indices = [
        i for i, label in enumerate(targets) if label in args.class_to_replace
    ]

    test_retain_dataset = Subset(test_loader.dataset, retain_indices)
    test_forget_dataset = Subset(test_loader.dataset, forget_indices)

    test_retain_loader = DataLoader(
        test_retain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    test_forget_loader = DataLoader(
        test_forget_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print("test")
    _, test_retain_acc = test_epoch(device, model, test_retain_loader, criterion)
    _, test_forget_acc = test_epoch(device, model, test_forget_loader, criterion)

    print(
        f"Linear Probing Results:\n"
        f"Train - Retain acc: {retain_acc:.2f}%, Forget acc: {forget_acc:.2f}%\n"
        f"Test  - Retain acc: {test_retain_acc:.2f}%, Forget acc: {test_forget_acc:.2f}%"
    )

    if args.wandb:
        logger.log(
            {
                "LP train retain acc": retain_acc,
                "LP train forget acc": forget_acc,
                "LP test retain acc": test_retain_acc,
                "LP test forget acc": test_forget_acc,
            }
        )

    evaluation_result["LP accuracy"] = {
        "LP train retain acc": retain_acc,
        "LP train forget acc": forget_acc,
        "LP test retain acc": test_retain_acc,
        "LP test forget acc": test_forget_acc,
    }

    if not args.no_save:
        save_evals(evaluation_result, args)

    if args.wandb:
        logger.finish()


def train_epoch(args, device, model, trainloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track Loss & Accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
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
