import torch
import numpy as np
import random
import os
import shutil


def set_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(state, is_SA_best, save_path, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, "best_state_dict.pth.tar"))


def load_checkpoint(device, save_path, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None
