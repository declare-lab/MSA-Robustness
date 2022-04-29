import os
import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
# from src.data_loader import *

from config import get_config
from data_loader import get_loader

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
# parser.add_argument('--data_path', type=str, default='data',
#                     help='path for storing the dataset')
parser.add_argument('--data_path', type=str, default='/data/yingting/mult_data',
                    help='path for storing the dataset')
########
parser.add_argument('--use_bert', action='store_true',
                    help='use bert as the language encoder')
########

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')
parser.add_argument('--activation', type=str, default='relu')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')

parser.add_argument('--train_method', type=str, default='missing', 
                        help='one of {missing, g_noise}, missing means set to zero noise, g_noise means set to Gaussian Noise')
parser.add_argument('--train_changed_modal', type=str, default='language', 
                        help='Control the modality of change during training. one of {language, video, audio}')
parser.add_argument('--train_changed_pct', type=float, default=0, 
                        help='Control the percentage of change during training')

parser.add_argument('--test_method', type=str, default='missing',  
                        help='one of {missing, g_noise}, missing means set to zero noise, g_noise means set to Gaussian Noise')
parser.add_argument('--test_changed_modal', type=str, default='language', 
                        help='Control the modality of change during testing. one of {language, video, audio}')
parser.add_argument('--test_changed_pct', type=float, default=0, 
                        help='Control the percentage of change during testing')

#Distinguish between eval and test
parser.add_argument('--is_test', action='store_true', help='Distinguish between eval and test')
#######################################################################################################

args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

# train_data = get_data(args, dataset, 'train')
# valid_data = get_data(args, dataset, 'valid')
# test_data = get_data(args, dataset, 'test')

# train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
# valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

#########################
train_config = get_config(dataset, mode='train', batch_size=args.batch_size, use_bert=args.use_bert)
valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size, use_bert=args.use_bert)
test_config = get_config(dataset, mode='test',  batch_size=args.batch_size, use_bert=args.use_bert)

train_loader = get_loader(args,train_config, shuffle=True)
valid_loader = get_loader(args,valid_config,shuffle=False)
test_loader = get_loader(args,test_config, shuffle=False)
########################


print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
# hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
# hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
# hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')
#####
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_config.lav_dim
if hyp_params.use_bert:
    hyp_params.orig_d_l = 768
# hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_config.lav_len
#####


if __name__ == '__main__':
    # test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)

    if hyp_params.is_test:
        print("--------------------is_test")
        test_loss = train.test_initiate(hyp_params, train_loader, valid_loader, test_loader)
    else:
        test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)

    # number = 0
    # for lr in [1e-3,8e-4]:
    #     for optim in ['Adam', 'RMSprop']:
    #         hyp_params.lr = lr
    #         hyp_params.optim = optim
    #         print(f"{number}=============== lr :{lr} ===== optim : {optim} =======================")
    #         test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
    #         number = number + 1

        # for optim in ['Adam', 'RMSprop']:
        #
        #[1e-3,1e-4,5e-4,8e-4,9e-4,2e-3],Adam 从这堆中选出来8e-4是效果最优的
        #['Adam', 'RMSprop']
        #目前来看，针对Mult MOSI, lr=8e-4, optim=RMSprop是最好的结果


