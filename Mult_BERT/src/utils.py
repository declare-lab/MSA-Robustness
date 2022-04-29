import torch
import os
from src.dataset import Multimodal_Datasets

from torch.utils.data import DataLoader
import numpy as np

def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.dataset


def save_model(args, model, name=''):
    name = save_load_name(args, name)

    if args.train_method=="missing":
        save_mode = f'0'
    elif args.train_method=="g_noise":
        save_mode = f'N'
    elif args.train_method=="hybird":
        save_mode = f'H'
    else:
        raise
    save_dir = f'checkpoints/{args.dataset}/best_{int(args.train_changed_pct*100)}%{args.train_changed_modal[0].upper()}={save_mode}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("-------------------------save_dir")
    print(save_dir)
    torch.save(model, f'{save_dir}/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)

    if args.train_method=="missing":
        save_mode = f'0'
    elif args.train_method=="g_noise":
        save_mode = f'N'
    elif args.train_method=="hybird":
        save_mode = f'H'
    else:
        raise
    save_dir = f'checkpoints/{args.dataset}/best_{int(args.train_changed_pct*100)}%{args.train_changed_modal[0].upper()}={save_mode}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("-------------------------save_dir")
    print(save_dir)
    model = torch.load(f'{save_dir}/{name}.pt')
    return model
