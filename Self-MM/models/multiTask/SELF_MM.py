# self supervised multimodal multi-task learning network
import os
import sys
import random
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subNets.BertTextEncoder import BertTextEncoder

__all__ = ['SELF_MM']

class SELF_MM(nn.Module):
    def __init__(self, args):
        super(SELF_MM, self).__init__()
        self.args = args
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

    def forward(self,text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text = self.text_model(text)[:,0,:]

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)

        ###############################################################
        #
        #TRAIN
        #
        ###############################################################
        if self.args.is_train:
            pct = self.args.train_changed_pct
            modal = self.args.train_changed_modal
            if modal == 'language':
                utterance = text
            elif modal == 'video':
                utterance = video
            elif modal == 'audio':
                utterance = audio
            else:
                print("Wrong modal!")
                exit()
            if self.args.train_method == 'missing':      # set modality to 0
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * 0
            elif self.args.train_method == 'g_noise':   # set modality to Noise
                noise = torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float().to(self.args.device)
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * noise[i]
            elif self.args.train_method == 'hybird':   # set half modality to 0, half modality to Noise
                noise = torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float().to(self.args.device)
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
                video = utterance
            elif modal == 'audio':
                audio = utterance
            else:
                print("Wrong modal!")
                exit()

        ###############################################################
        #
        #TEST
        #
        ###############################################################
        if self.args.is_test:
            test_modal = self.args.test_changed_modal
            test_pct = self.args.test_changed_pct
            if test_modal == 'language':
                utterance = text
            elif test_modal == 'video':
                utterance = video
            elif test_modal == 'audio':
                utterance = audio
            else:
                print("Wrong test_modal!")
                exit()
            if self.args.test_method == 'missing':
                for i, _ in enumerate(utterance):
                    rand_num = torch.rand(1)
                    if rand_num < test_pct:
                        utterance[i] = utterance[i] * 0
            elif self.args.test_method == 'g_noise':
                noise = torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float().to(self.args.device)
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
                video = utterance
            elif test_modal == 'audio':
                audio = utterance
            else:
                print("Wrong test_modal!")
                exit()
        
        # fusion
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1
