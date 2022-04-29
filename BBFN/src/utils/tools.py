import torch
import os


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    # return name + '_' + args.model
    return name


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    # if not os.path.exists('pre_trained_models'):
    #     os.mkdir('pre_trained_models')
    # torch.save(model, f'pre_trained_models/{name}.pt')

    # save_dir = f'pre_trained_models/{args.data}/best_{int(args.train_changed_pct*100)}%{args.train_changed_modal[0].upper()}={0 if args.train_method=="missing" else "N"}/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    if args.train_method=="missing":
        save_mode = f'0'
    elif args.train_method=="g_noise":
        save_mode = f'N'
    elif args.train_method=="hybird":
        save_mode = f'H'
    else:
        raise
    save_dir = f'pre_trained_models/{args.data}/best_{int(args.train_changed_pct*100)}%{args.train_changed_modal[0].upper()}={save_mode}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("-------------------------save_dir")
    print(save_dir)

    torch.save(model, f'{save_dir}/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    # model = torch.load(f'pre_trained_models/{name}.pt')

    # save_dir = f'pre_trained_models/{args.data}/best_{int(args.train_changed_pct*100)}%{args.train_changed_modal[0].upper()}={0 if args.train_method=="missing" else "N"}/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    if args.train_method=="missing":
        save_mode = f'0'
    elif args.train_method=="g_noise":
        save_mode = f'N'
    elif args.train_method=="hybird":
        save_mode = f'H'
    else:
        raise
    save_dir = f'pre_trained_models/{args.data}/best_{int(args.train_changed_pct*100)}%{args.train_changed_modal[0].upper()}={save_mode}/'
    print("-------------------------save_dir")
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = torch.load(f'{save_dir}/{name}.pt')
    
    return model

def to_gpu(x, on_cpu=False, gpu_id=None):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id)
    return x

def to_cpu(x):
    """Variable => Tensor"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data
