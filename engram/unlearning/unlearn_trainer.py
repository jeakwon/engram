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
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    random_split,
    ConcatDataset,
    Subset,
)
import engram.unlearning.unlearn_trainer as unlearn_trainer
import utils
import numpy as np
import pandas as pd
import time

# import pruner
from trainer import validate

from surgical_plugins.cluster import (
    get_features,
    get_distance,
    get_fs,
    get_fs_dist_only,
)
from surgical_plugins.overlap import calculate_FC, compute_diagonal_fisher_information


def main():
    start_rte = time.time()
    args = arg_parser.parse_args()

    base_name = f"{args.arch}-{args.dataset}-{args.unlearn}"
    logger = wandb_init(args)
    files_to_save = []

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    args.save_dir = f"result/unlearn/{args.unlearn}"

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    # prepare dataset
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        (
            model,
            train_loader_full,
            val_loader,
            test_loader,
            marked_loader,
            train_idx,
        ) = utils.setup_model_dataset(args)
    elif args.dataset == "TinyImagenet":
        args.data_dir = "/data/image_data/tiny-imagenet-200/"
        (
            model,
            train_loader_full,
            val_loader,
            test_loader,
            marked_loader,
        ) = utils.setup_model_dataset(args)
    model.cuda()
    rich_print(args)

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)

    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        print(len(forget_dataset))
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        print(len(retain_dataset))
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )

    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unique_classes, counts = np.unique(forget_dataset.targets, return_counts=True)
    class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
    print("forget set: ")
    print(class_counts)
    print("retain set: ", len(retain_dataset))

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    """
    print('val dataset:')
    for i, (image, target) in enumerate(val_loader):
        print(target)

    print('test dataset:')   
    for i, (image, target) in enumerate(test_loader):
        print(target)
    """

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        print("check 3, which model to load: ", args.mask)
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)
            print("check 4: model loaded!")

        # calculate_FC(model, retain_dataset, forget_dataset, args)

        print(
            f"-------------------Get unlearning method: {args.unlearn}-------------------"
        )
        start_unlearn = time.time()
        if (
            args.unlearn == "original"
            or args.unlearn == "seq_mix"
            or args.unlearn == "mix"
        ):
            pass
        else:
            unlearn_method = unlearn_trainer.get_unlearn_method(args.unlearn)
            if args.unlearn == "SCRUB":
                model_s = copy.deepcopy(model)
                model_t = copy.deepcopy(model)
                module_list = nn.ModuleList([model_s, model_t])
                unlearn_method(unlearn_data_loaders, module_list, criterion, args)
                model = module_list[0]
            else:
                unlearn_method(unlearn_data_loaders, model, criterion, args)
            if args.no_save:
                pass
            else:
                unlearn_trainer.impl.save_unlearn_checkpoint(model, None, args)
                print("check 5: unlearned model saved!")

    end_rte = time.time()
    print(
        f"Overall time taken for unlearning & preparation: {end_rte - start_rte:.3f}s"
    )
    print(f"Time taken for unlearning only: {end_rte - start_unlearn:.3f}s")
    logger.log({"unlearn_time": end_rte - start_unlearn})
    logger.log({"overall_time (unlearning & preparation)": end_rte - start_rte})

    print("-------------------Start acc evaluation-------------------")
    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            print(name)
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        if mem_fs_split:
            logger.log(
                {
                    "forget acc": accuracy["forget"],
                    "retain acc": accuracy["retain"],
                    "val acc": accuracy["val"],
                    "test acc": accuracy["test"],
                    "high mem acc": accuracy["high_mem"],
                    "mid mem acc": accuracy["mid_mem"],
                    "low mem acc": accuracy["low_mem"],
                }
            )
        elif fine_overlap:
            logger.log(
                {
                    "forget acc": accuracy["forget"],
                    "retain acc": accuracy["retain"],
                    "val acc": accuracy["val"],
                    "test acc": accuracy["test"],
                    "high des acc": accuracy["high_des"],
                    "mid des acc": accuracy["mid_des"],
                    "low des acc": accuracy["low_des"],
                }
            )
        elif proxy_fs_split:
            logger.log(
                {
                    "forget acc": accuracy["forget"],
                    "retain acc": accuracy["retain"],
                    "val acc": accuracy["val"],
                    "test acc": accuracy["test"],
                    "high proxy acc": accuracy["high_proxy"],
                    "mid proxy acc": accuracy["mid_proxy"],
                    "low proxy acc": accuracy["low_proxy"],
                }
            )
        else:
            logger.log(
                {
                    "forget acc": accuracy["forget"],
                    "retain acc": accuracy["retain"],
                    "val acc": accuracy["val"],
                    "test acc": accuracy["test"],
                }
            )

        evaluation_result["accuracy"] = accuracy
        if args.no_save:
            pass
        else:
            unlearn_trainer.impl.save_unlearn_checkpoint(model, evaluation_result, args)

    print("-------------------Start MIA evaluation-------------------")
    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain (shadow train - label 1)
        out of distribution: test (shadow train - label 0)
        target: (, forget)"""
    MIA_forget_efficacy = True
    if MIA_forget_efficacy:
        if "SVC_MIA_forget_efficacy" not in evaluation_result:
            test_len = len(test_loader.dataset)
            forget_len = len(forget_dataset)
            retain_len = len(retain_dataset)

            utils.dataset_convert_to_test(retain_dataset, args)
            utils.dataset_convert_to_test(forget_loader, args)
            utils.dataset_convert_to_test(test_loader, args)

            shadow_train = torch.utils.data.Subset(
                retain_dataset, list(range(test_len))
            )
            shadow_train_loader = torch.utils.data.DataLoader(
                shadow_train, batch_size=args.batch_size, shuffle=False
            )

            evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
                shadow_train=shadow_train_loader,
                shadow_test=test_loader,
                target_train=None,
                target_test=forget_loader,
                model=model,
            )
            if args.no_save:
                pass
            else:
                unlearn_trainer.impl.save_unlearn_checkpoint(
                    model, evaluation_result, args
                )
            logger.log(
                {
                    "SVC_MIA_forget_efficacy": evaluation_result[
                        "SVC_MIA_forget_efficacy"
                    ]
                }
            )

    """training privacy MIA:
        in distribution: retain
        out of distribution: test
        target: (retain, test)"""
    MIA_training_privacy = False
    if MIA_training_privacy:
        if "SVC_MIA_training_privacy" not in evaluation_result:
            test_len = len(test_loader.dataset)
            retain_len = len(retain_dataset)
            num = test_len // 2

            utils.dataset_convert_to_test(retain_dataset, args)
            utils.dataset_convert_to_test(forget_loader, args)
            utils.dataset_convert_to_test(test_loader, args)

            shadow_train = torch.utils.data.Subset(retain_dataset, list(range(num)))
            target_train = torch.utils.data.Subset(
                retain_dataset, list(range(num, retain_len))
            )
            shadow_test = torch.utils.data.Subset(test_loader.dataset, list(range(num)))
            target_test = torch.utils.data.Subset(
                test_loader.dataset, list(range(num, test_len))
            )

            shadow_train_loader = torch.utils.data.DataLoader(
                shadow_train, batch_size=args.batch_size, shuffle=False
            )
            shadow_test_loader = torch.utils.data.DataLoader(
                shadow_test, batch_size=args.batch_size, shuffle=False
            )

            target_train_loader = torch.utils.data.DataLoader(
                target_train, batch_size=args.batch_size, shuffle=False
            )
            target_test_loader = torch.utils.data.DataLoader(
                target_test, batch_size=args.batch_size, shuffle=False
            )

            evaluation_result["SVC_MIA_training_privacy"] = evaluation.SVC_MIA(
                shadow_train=shadow_train_loader,
                shadow_test=shadow_test_loader,
                target_train=target_train_loader,
                target_test=target_test_loader,
                model=model,
            )
            if args.no_save:
                pass
            else:
                unlearn_trainer.impl.save_unlearn_checkpoint(
                    model, evaluation_result, args
                )
            logger.log(
                {
                    "SVC_MIA_training_privacy": evaluation_result[
                        "SVC_MIA_training_privacy"
                    ]
                }
            )

    if args.no_save:
        pass
    else:
        unlearn_trainer.impl.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
