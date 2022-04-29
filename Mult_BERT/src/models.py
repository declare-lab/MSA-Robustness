import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder

###########
from transformers import BertModel, BertConfig
from encoders import LanguageEmbeddingLayer
# from torch.nn.functional import relu
import numpy as np
import random
#######


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        # self.d_l, self.d_a, self.d_v = 40, 40, 40
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        ############################
        self.hyp_params = hyp_params
        ############################

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        ############## use bert #####################
        self.embedding = LanguageEmbeddingLayer(self.hyp_params)
        #############################################

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)


    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    # def forward(self, x_l, x_a, x_v):
    def forward(self, is_train, x_l, x_v, x_a, y, l, bert_sent, bert_sent_type, bert_sent_mask):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        # x_l: (batch_size, 102)
        # x_a: (batch_size, 102, 74)
        # x_v: (batch_size, 102, 47)

        # x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training) #(batch_size, 300, 50)
        # x_a = x_a.transpose(1, 2)                 #(batch_size, 5, 50)    
        # x_v = x_v.transpose(1, 2)                 #(batch_size, 20, 50)  

        x_a = x_a.transpose(0,1)
        x_v = x_v.transpose(0,1)
       
        ################
        def _pad_seq(video, acoustic, lengths):
            pld = (0, 0, 0, 0, 1, 1)   #在第0个维度，1 + 28 + 1，前面加1且后面加1
            pad_video = F.pad(video, pld, "constant", 0.0)
            pad_acoustic = F.pad(acoustic, pld, "constant", 0.0)
            lengths = lengths + 2
            return pad_video, pad_acoustic, lengths

        ## use bert
        proj_x_l = self.embedding(x_l, l, bert_sent, bert_sent_type, bert_sent_mask)
        x_v, x_a, _ = _pad_seq(x_v, x_a, l)
        
        x_a = x_a.transpose(0, 1)
        x_v = x_v.transpose(0, 1)
        x_a = x_a.transpose(1, 2)  #(batch_size, 74, 102)
        x_v = x_v.transpose(1, 2)  #(batch_size, 47, 102)
        proj_x_l = F.dropout(proj_x_l.transpose(1,2), p=self.embed_dropout, training=self.training)

        #####################################################
        # Control experiment
        #####################################################
        # print("====== language multiply missing ======")
        # proj_x_l = proj_x_l * 0

        # print("----language multiply missing 30%-------")
        # for i, _ in enumerate(proj_x_l):
        #     rand_num = torch.rand(1)
        #     if rand_num < 0.3:
        #         proj_x_l[i] = proj_x_l[i] * 0

        # print("--------------language multiply noise-------------------")
        # noise = torch.from_numpy(np.random.normal(0,1,proj_x_l.size()[2])).float().to(proj_x_l.device)
        # proj_x_l = proj_x_l * noise

        # print("-------------language multiply noise  30%---------------------")
        # noise = torch.from_numpy(np.random.normal(0,1,proj_x_l.size()[0])).float().to(proj_x_l.device)
        # sample_num = int(len(proj_x_l) * 0.3)
        # sample_list = [i for i in range(len(proj_x_l))]
        # sample_list = random.sample(sample_list, sample_num)
        # for i in sample_list:
        #     proj_x_l[i] = proj_x_l[i] * noise[i]
        ##################### TRAIN #########################
        if is_train:
            pct = self.hyp_params.train_changed_pct
            modal = self.hyp_params.train_changed_modal
            if modal == 'language':
                utterance = proj_x_l
            elif modal == 'video':
                utterance = x_v
            elif modal == 'audio':
                utterance = x_a
            else:
                print("Wrong modal!")
                exit()
            if self.hyp_params.train_method == 'missing':      # set modality to 0
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * 0
            elif self.hyp_params.train_method == 'g_noise':   # set modality to Noise
                noise = torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float().to(proj_x_l.device)
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * noise[i]
            elif self.hyp_params.train_method == 'hybird':   # set half modality to 0, half modality to Noise
                noise = torch.from_numpy(np.random.normal(0,1,utterance.size()[1])).float().to(proj_x_l.device)
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list_0 = random.sample(sample_list, sample_num)
                sample_list_new = list(set(sample_list).difference(set(sample_list_0)))
                sample_list_N = random.sample(sample_list_new, sample_num)
                for i in sample_list_0:
                    utterance[i] = utterance[i] * 0
                for i in sample_list_N:
                    utterance[i] = utterance[i] * noise[i]
            else:
                print("Wrong method!")
                exit()

            if modal == 'language':
                proj_x_l = utterance
            elif modal == 'video':
                x_v = utterance
            elif modal == 'audio':
                x_a = utterance
            else:
                print("Wrong modal!")
                exit()

        ###################### TEST #########################
        if self.hyp_params.is_test:
            test_modal = self.hyp_params.test_changed_modal
            test_pct = self.hyp_params.test_changed_pct
            if test_modal == 'language':
                utterance = proj_x_l
            elif test_modal == 'video':
                utterance = x_v
            elif test_modal == 'audio':
                utterance = x_a
            else:
                print("Wrong test_modal!")
                exit()
            if self.hyp_params.test_method == 'missing':
                for i, _ in enumerate(utterance):
                    rand_num = torch.rand(1)
                    if rand_num < test_pct:
                        utterance[i] = utterance[i] * 0
            elif self.hyp_params.test_method == 'g_noise':
                noise = torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float().to(proj_x_l.device)
                sample_num = int(len(utterance) * test_pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * noise[i]
            else:
                print("Wrong method!")
                exit()

            if test_modal == 'language':
                proj_x_l = utterance
            elif test_modal == 'video':
                x_v = utterance
            elif test_modal == 'audio':
                x_a = utterance
            else:
                print("Wrong test_modal!")
                exit()
        ################### END #############################

        ##############
        # Project the textual/visual/audio features
        proj_x_l = self.proj_l(proj_x_l)
        # proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output, last_hs