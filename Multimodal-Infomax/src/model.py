import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet

from transformers import BertModel, BertConfig
from utils.tools import to_gpu

class MMIM(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp
        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        self.text_enc = LanguageEmbeddingLayer(hp)
        self.visual_enc = RNNEncoder(
            in_size = hp.d_vin,
            hidden_size = hp.d_vh,
            out_size = hp.d_vout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size = hp.d_ain,
            hidden_size = hp.d_ah,
            out_size = hp.d_aout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )

        # For MI maximization
        self.mi_tv = MMILB(
            x_size = hp.d_tout,
            y_size = hp.d_vout,
            mid_activation = hp.mmilb_mid_activation,
            last_activation = hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size = hp.d_tout,
            y_size = hp.d_aout,
            mid_activation = hp.mmilb_mid_activation,
            last_activation = hp.mmilb_last_activation
        )

        if hp.add_va:
            self.mi_va = MMILB(
                x_size = hp.d_vout,
                y_size = hp.d_aout,
                mid_activation = hp.mmilb_mid_activation,
                last_activation = hp.mmilb_last_activation
            )

        dim_sum = hp.d_aout + hp.d_vout + hp.d_tout

        # CPC MI bound
        self.cpc_zt = CPC(
            x_size = hp.d_tout, # to be predicted
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size = hp.d_vout,
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )
        self.cpc_za = CPC(
            x_size = hp.d_aout,
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )

        # Trimodal Settings
        self.fusion_prj = SubNet(
            # in_size = dim_sum,
            in_size = hp.d_tout,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )
            
    def forward(self, is_train, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None, mem=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """

        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        text = enc_word[:,0,:] # (batch_size, emb_size)

        acoustic = self.acoustic_enc(acoustic, a_len)
        visual = self.visual_enc(visual, v_len)
        ###############################################################
        #
        #TRAIN
        #
        ###############################################################
        # print("==================")
        # print("text    :", text.size())
        # print("visual  :", visual.size())
        # print("acoustic:", acoustic.size())
        # exit()
        if is_train:
            pct = self.hp.train_changed_pct
            modal = self.hp.train_changed_modal
            if modal == 'language':
                utterance = text
            elif modal == 'video':
                utterance = visual
            elif modal == 'audio':
                utterance = acoustic
            else:
                print("Wrong modal!")
                exit()
            if self.hp.train_method == 'missing':      # set modality to 0
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * 0
            elif self.hp.train_method == 'g_noise':   # set modality to Noise
                noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float())
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * noise[i]
            elif self.hp.train_method == 'hybird':   # set half modality to 0, half modality to Noise
                noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float())
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
                text = utterance
            elif modal == 'video':
                visual = utterance
            elif modal == 'audio':
                acoustic = utterance
            else:
                print("Wrong modal!")
                exit()

        ###############################################################
        #
        #TEST
        #
        ###############################################################
        if self.hp.is_test:
            test_modal = self.hp.test_changed_modal
            test_pct = self.hp.test_changed_pct
            if test_modal == 'language':
                utterance = text
            elif test_modal == 'video':
                utterance = visual
            elif test_modal == 'audio':
                utterance = acoustic
            else:
                print("Wrong test_modal!")
                exit()
            if self.hp.test_method == 'missing':
                for i, _ in enumerate(utterance):
                    rand_num = torch.rand(1).item()
                    if rand_num < test_pct:
                        utterance[i] = utterance[i] * 0
            elif self.hp.test_method == 'g_noise':
                noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float())
                sample_num = int(len(utterance) * test_pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * noise[i]
            else:
                print("Wrong method!")
                exit()

            if test_modal == 'language':
                text = utterance
            elif test_modal == 'video':
                visual = utterance
            elif test_modal == 'audio':
                acoustic = utterance
            else:
                print("Wrong test_modal!")
                exit()

        ##############
        # visual = torch.zeros_like(visual)
        # acoustic = torch.zeros_like(acoustic)
        # visual = visual * 0
        # acoustic = acoustic * 0
        ###############
        if y is not None:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])

            # for ablation use
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic, labels=y, mem=mem['va'])
        else:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)

            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic)

        # Linear proj and pred
        # fusion, preds = self.fusion_prj(torch.cat([text, acoustic, visual], dim=1))
        ######text only
        # pad_sen = torch.zeros(len(text), len(acoustic[0]) + len(visual[0])).to(text.device)
        # fusion, preds = self.fusion_prj(torch.cat([text,pad_sen],dim=1))
        fusion, preds = self.fusion_prj(text)
        ###### bi v and a
        # pad_sen = torch.zeros(len(text), len(text[0])).to(text.device)
        # fusion, preds = self.fusion_prj(torch.cat([pad_sen,acoustic, visual],dim=1))

        nce_t = self.cpc_zt(text, fusion)
        nce_v = self.cpc_zv(visual, fusion)
        nce_a = self.cpc_za(acoustic, fusion)

        ##################
        nce_v = torch.zeros_like(nce_v)
        nce_a = torch.zeros_like(nce_a)
        ##################
        
        nce = nce_t + nce_v + nce_a

        pn_dic = {'tv':tv_pn, 'ta':ta_pn, 'va': va_pn if self.add_va else None}
        lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)
        H = H_tv + H_ta + (H_va if self.add_va else 0.0)
        # print("========END one")

        return lld, nce, preds, pn_dic, H


        ##############################################################
        #
        #Add noise
        #
        ##############################################################

        # print("====== language multiply missing ======")
        # text = text * 0

        # print("----language multiply missing 30%-------")
        # for i, _ in enumerate(text):
        #     rand_num = torch.rand(1)
        #     if rand_num < 0.3:
        #         text[i] = text[i] * 0

        # print("--------------language multiply noise-------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1,text.size()[1])).float())
        # text = text * noise

        # print("-------------language multiply noise  30%---------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1,text.size()[0])).float())
        # sample_num = int(len(text) * 0.3)
        # sample_list = [i for i in range(len(text))]
        # sample_list = random.sample(sample_list, sample_num)
        # for i in sample_list:
        #     text[i] = text[i] * noise[i]
        ############################################
        # print("====== visual multiply missing ======")
        # visual = visual * 0

        # print("----visual multiply missing 30%-------")
        # for i, _ in enumerate(visual):
        #     rand_num = torch.rand(1)
        #     if rand_num < 0.3:
        #         visual[i] = visual[i] * 0

        # print("--------------video multiply noise-------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1.5,visual.size()[1])).float())
        # visual = visual * noise

        # print("-------------video multiply noise  30%---------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1,visual.size()[0])).float())
        # sample_num = int(len(visual) * 0.3)
        # sample_list = [i for i in range(len(visual))]
        # sample_list = random.sample(sample_list, sample_num)
        # for i in sample_list:
        #     visual[i] = visual[i] * noise[i]
        ##############################################
        # print("--------------audio multiply noise-------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1.5,acoustic.size()[1])).float())
        # acoustic = acoustic * noise

        # print("====== acoustic multiply missing ======")
        # acoustic = acoustic * 0

        # print("----acoustic multiply missing 30%-------")
        # for i, _ in enumerate(acoustic):
        #     rand_num = torch.rand(1)
        #     if rand_num < 0.3:
        #         acoustic[i] = acoustic[i] * 0

        # print("-------------acoustic multiply noise  30%---------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1,acoustic.size()[0])).float())
        # sample_num = int(len(acoustic) * 0.3)
        # sample_list = [i for i in range(len(acoustic))]
        # sample_list = random.sample(sample_list, sample_num)
        # for i in sample_list:
        #     acoustic[i] = acoustic[i] * noise[i]
        ##############################################################