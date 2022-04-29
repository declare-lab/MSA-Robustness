import torch
import torch.nn as nn
import numpy as np
from random import random
from sklearn.metrics import f1_score, classification_report, accuracy_score

from config import get_config, activation_dict
from data_loader import get_loader
from solver import Solver
from utils import to_gpu
import models

import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Setting random seed
    # random_name = str(random())
    # random_seed = 336   
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(random_seed)
    
    # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    # print(test_config)

    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)

    # load model
    model = getattr(models, test_config.model)(test_config)
    # if test_config.data == 'mosi':
    #     path = f'checkpoints/best_model_mosi.std'

        # model.load_state_dict(torch.load(
        #                 f'checkpoints/model_2021-11-14_00:02:34.std'))  # mosi
        # model.load_state_dict(torch.load(f'checkpoints/best_model_mosi.std'))
    # elif test_config.data == 'mosei':
    #     path = f'checkpoints/best_model_mosei.std'
        # path = f'checkpoints/mosei/best_30%L=0_train/best_model_mosei.std'
        # path = f'checkpoints/mosei/best_30%L=N_train/best_model_mosei.std'

        # model.load_state_dict(torch.load(
        #                 f'checkpoints/model_2021-11-14_14:14:10.std'))  # mosei
        # model.load_state_dict(torch.load(f'checkpoints/best_model_mosei.std'))


    # Solver is a wrapper for model traiing and testing
    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=False, model=model)

    # Build the model
    solver.build()
    solver.criterion = nn.MSELoss(reduction="mean")

    # test the model (test scores will be returned based on dev performance)
    solver.eval(mode="test", to_print=True)

