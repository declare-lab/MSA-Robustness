import os
import torch
import logging
import argparse
import random
import numpy as np

from models.AMIO import AMIO
from trains.ATIO import ATIO

from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_tune', type=bool, default=False,
                        help='tune parameters ?')
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression / classification')
    parser.add_argument('--modelName', type=str, default='self_mm',
                        help='support self_mm')
    parser.add_argument('--datasetName', type=str, default='sims',
                        help='support mosi/mosei/sims')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=list, default=[1],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')
    ##################################################################################################
    #Control training method
    parser.add_argument('--train_method', type=str, default='missing', 
                            help='one of {missing, g_noise}, missing means set to zero noise, g_noise means set to Gaussian Noise')
    #Control the modality of change during training
    parser.add_argument('--train_changed_modal', type=str, default='language', help='one of {language, video, audio}')
    #Control the percentage of change during training
    parser.add_argument('--train_changed_pct', type=float, default=0.3, help='Control the percentage of change during training')
    

    #Control testing method
    parser.add_argument('--test_method', type=str, default='missing',  
                            help='one of {missing, g_noise}, missing means set to zero noise, g_noise means set to Gaussian Noise')
    #Control the modality of change during testing
    parser.add_argument('--test_changed_modal', type=str, default='language', help='one of {language, video, audio}')
    #Control the percentage of change during training
    parser.add_argument('--test_changed_pct', type=float, default=0.3, help='Control the percentage of change during testing')

    #Distinguish between eval and test
    parser.add_argument('--is_test', action='store_true', help='Distinguish between eval and test')
    parser.add_argument('--is_train', action='store_true', help='Distinguish between eval and test')
    #######################################################################################################
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # logger = set_log(args)
    args.model_save_path = os.path.join(args.model_save_dir,\
                                        f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth')
    
    args.seeds = [1111,1112, 1113, 1114, 1115]
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device

    seed = args.seeds[0]

    # load config
    if args.train_mode == "regression":
        config = ConfigRegression(args)
    args = config.get_config()
    setup_seed(seed)
    args.seed = seed
    
    # load model
    model = AMIO(args).to(device)
    #######################
    # save_dir = f'results/models/{args.datasetName}/best_{int(args.train_changed_pct*100)}%{args.train_changed_modal[0].upper()}={0 if args.train_method=="missing" else "N"}/'
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
    save_dir = f'results/models/{args.datasetName}/best_{int(args.train_changed_pct*100)}%{args.train_changed_modal[0].upper()}={save_mode}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    args.model_save_path = os.path.join(save_dir, f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth')
    #######################
    model.load_state_dict(torch.load(args.model_save_path))

    # load data
    dataloader = MMDataLoader(args)

    atio = ATIO().getTrain(args)

    results = atio.do_test(model, dataloader['test'], mode="TEST")

    print("---------------------------------")
    print("results:", results)