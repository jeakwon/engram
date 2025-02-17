import copy
import os
from collections import OrderedDict
from rich import print as rich_print
import matplotlib.pyplot as plt
import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, Subset
import unlearn
import utils
import numpy as np
import pandas as pd
import time

from trainer import validate
from unlearn.impl import wandb_init, wandb_finish


def main():
    start_rte = time.time()
    args = arg_parser.parse_args()

    # 가정: args.unlearn_step = None, args.sequential = False
    args.wandb_group_name = f"{args.arch}_{args.dataset}_{args.unlearn}"
    logger = wandb_init(args)
    args.save_dir = f"result/unlearn/{args.unlearn}"
    os.makedirs(args.save_dir, exist_ok=True)

    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 및 모델 준비 (cifar10, cifar100, TinyImagenet 지원)
    if args.dataset in ["cifar10", "cifar100"]:
        model, train_loader_full, val_loader, test_loader, marked_loader, train_idx = (
            utils.setup_model_dataset(args)
        )
    elif args.dataset == "TinyImagenet":
        args.data_dir = "/data/image_data/tiny-imagenet-200/"
        model, train_loader_full, val_loader, test_loader, marked_loader = (
            utils.setup_model_dataset(args)
        )
    model.cuda()
    rich_print(args)

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    # marked_loader를 기반으로 forget(삭제)와 retain(유지) 데이터셋 생성
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    try:
        marked = forget_dataset.targets < 0
        forget_dataset.data = forget_dataset.data[marked]
        forget_dataset.targets = -forget_dataset.targets[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        print("Forget dataset size:", len(forget_dataset))

        retain_dataset = copy.deepcopy(marked_loader.dataset)
        marked = retain_dataset.targets >= 0
        retain_dataset.data = retain_dataset.data[marked]
        retain_dataset.targets = retain_dataset.targets[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        print("Retain dataset size:", len(retain_dataset))

        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    except:
        # 예를 들어, ImageFolder처럼 targets 대신 imgs를 사용할 경우
        marked = forget_dataset.targets < 0
        forget_dataset.imgs = forget_dataset.imgs[marked]
        forget_dataset.targets = -forget_dataset.targets[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        print("Forget dataset size:", len(forget_dataset))

        retain_dataset = copy.deepcopy(marked_loader.dataset)
        marked = retain_dataset.targets >= 0
        retain_dataset.imgs = retain_dataset.imgs[marked]
        retain_dataset.targets = retain_dataset.targets[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        print("Retain dataset size:", len(retain_dataset))

        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )

    print(f"Number of retain dataset: {len(retain_dataset)}")
    print(f"Number of forget dataset: {len(forget_dataset)}")
    unique_classes, counts = np.unique(forget_dataset.targets, return_counts=True)
    class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
    print("Forget set class distribution:")
    print(class_counts)

    # unlearn_data_loaders 구성: retain, forget, val, test
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    # args.unlearn_step가 None이므로 원본 모델 파일 사용
    args.mask = f"assets/checkpoints/0{args.dataset}_original_{args.arch}_bs256_lr0.1_seed{args.seed}_epochs{args.epochs}.pth.tar"
    print("Load original model from:", args.mask)

    evaluation_result = None
    if args.resume:
        checkpoint = unlearn.impl.load_unlearn_checkpoint(model, device, args)
    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        print("Loading model from:", args.mask)
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)
            print("Model loaded successfully.")

        print(
            f"-------------------Unlearning method: {args.unlearn}-------------------"
        )
        start_unlearn = time.time()
        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        if args.unlearn == "SCRUB":
            model_s = copy.deepcopy(model)
            model_t = copy.deepcopy(model)
            module_list = nn.ModuleList([model_s, model_t])
            unlearn_method(unlearn_data_loaders, module_list, criterion, args)
            model = module_list[0]
        else:
            unlearn_method(unlearn_data_loaders, model, criterion, args)
        if not args.no_save:
            unlearn.impl.save_unlearn_checkpoint(model, None, args)
            print("Unlearned model saved.")
    end_rte = time.time()
    print(f"Overall time (unlearning & preparation): {end_rte - start_rte:.3f}s")
    print(f"Unlearning time: {end_rte - start_unlearn:.3f}s")
    logger.log({"unlearn_time": end_rte - start_unlearn})
    logger.log({"overall_time (unlearning & preparation)": end_rte - start_rte})

    # Accuracy 평가
    print("-------------------Start accuracy evaluation-------------------")
    if evaluation_result is None:
        evaluation_result = {}
    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            print("Evaluating on:", name)
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
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
            unlearn.impl.save_unlearn_checkpoint(model, evaluation_result, args)

    # MIA 평가 (forget efficacy)
    print("-------------------Start MIA evaluation-------------------")
    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        evaluation_result.pop(deprecated, None)
    MIA_forget_efficacy = True
    if MIA_forget_efficacy and "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)
        shadow_train = Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )
        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        if not args.no_save:
            unlearn.impl.save_unlearn_checkpoint(model, evaluation_result, args)
        logger.log(
            {"SVC_MIA_forget_efficacy": evaluation_result["SVC_MIA_forget_efficacy"]}
        )

    if not args.no_save:
        unlearn.impl.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
