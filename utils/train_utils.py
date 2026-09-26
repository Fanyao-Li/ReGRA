import os
import json
import torch
import random
import numpy as np
from datetime import datetime
from transformers import set_seed


def set_global_seed(seed=-1):
    if seed < 0:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_target_modules(keys):
    target_modules = []
    if 'q' in keys:
        target_modules.append("q_proj")
    if 'k' in keys:
        target_modules.append("k_proj")
    if 'v' in keys:
        target_modules.append("v_proj")
    if 'u' in keys:
        target_modules.append("up_proj")
    if 'd' in keys:
        target_modules.append("down_proj")
    if 'o' in keys:
        target_modules.append("o_proj")
    if 'g' in keys:
        target_modules.append("gate_proj")
    return target_modules

def get_trainable_params_numbers(model, path=None):
    all_param = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rate = {
        "trainable_params": trainable_params,
        "all_params": all_param,
        "trainable_ratio": trainable_params / all_param,
    }
    if path is not None:
        with open(path, 'w') as json_file:
            json.dump(rate, json_file, indent=4)
    return rate

def print_delta_time(start_time, logger=None):
    end_time = datetime.now()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    if logger is None:  
        print(f"start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"real  time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"delta time: {int(hours)} hours, {int(minutes)} minutes")
    else:
        logger.info(f"start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"real  time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"delta time: {int(hours)} hours, {int(minutes)} minutes")
    return int(hours), int(minutes)

def convert_lora_params_dtype(model, dtype="bf16"):
    if dtype == "fp32":
        real_dtype = torch.float32
    elif dtype == "bf16":
        real_dtype = torch.bfloat16
    elif dtype == "fp16":
        real_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported lora dtype: {dtype}.")
    
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.to(real_dtype)
            print(f"converted {name} to {real_dtype}")
