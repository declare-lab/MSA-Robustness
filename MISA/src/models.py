import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from utils import to_gpu
from utils import ReverseLayerF


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)



# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size


        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between

        if self.config.use_bert:

            # Initializing a BERT bert-base-uncased style configuration
            bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        else:
            self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)



        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size)) #config.hidden_size == 128
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))


        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))



        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))



        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))


        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, is_train, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        
        batch_size = lengths.size(0)
        
        if self.config.use_bert:
            bert_output = self.bertmodel(input_ids=bert_sent, 
                                         attention_mask=bert_sent_mask, 
                                         token_type_ids=bert_sent_type)      

            bert_output = bert_output[0]

            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

            utterance_text = bert_output
        else:
            # extract features from text modality
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)



        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        ###############################################################
        #
        #TRAIN
        #
        ###############################################################
        ###############SET L TO 0##########################
        # if is_train:
        #     sample_num = int(len(utterance_text) * 0.5)
        #     sample_list = [i for i in range(len(utterance_text))]
        #     sample_list = random.sample(sample_list, sample_num)
        #     for i in sample_list:
        #         utterance_text[i] = utterance_text[i] * 0
        ###############ADD NOISE TO L##########################
        # if is_train:
        #     noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance_video.size()[1])).float())
        #     sample_num = int(len(utterance_text) * 0.5)
        #     sample_list = [i for i in range(len(utterance_text))]
        #     sample_list = random.sample(sample_list, sample_num)
        #     for i in sample_list:
        #         utterance_text[i] = utterance_text[i] * noise[i]
        # if is_train:
        #     noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance_video.size()[1])).float())
        #     sample_num = int(len(utterance_text) * 0.15)
        #     sample_list = [i for i in range(len(utterance_text))]
        #     sample_list_0 = random.sample(sample_list, sample_num)
        #     sample_list_N = random.sample(sample_list, sample_num)
        #     #需要保证sample_list_0 sample_list_N没有交集吗
        #     for i in sample_list_0:
        #         utterance_text[i] = utterance_text[i] * 0
        #     for i in sample_list_N:
        #         utterance_text[i] = utterance_text[i] * noise[i]

        ###############################################################
        if is_train:
            pct = self.config.train_changed_pct
            modal = self.config.train_changed_modal
            if modal == 'language':
                utterance = utterance_text
            elif modal == 'video':
                utterance = utterance_video
            elif modal == 'audio':
                utterance = utterance_audio
            else:
                print("Wrong modal!")
                exit()
            if self.config.train_method == 'missing':      # set modality to 0
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * 0
            elif self.config.train_method == 'g_noise':   # set modality to Noise
                noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float())
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * noise[i]
            elif self.config.train_method == 'hybird':   # set half modality to 0, half modality to Noise
                noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance.size()[1])).float())
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
                print("Wrong train method!")
                exit()

            if modal == 'language':
                utterance_text = utterance
            elif modal == 'video':
                utterance_video = utterance
            elif modal == 'audio':
                utterance_audio = utterance
            else:
                print("Wrong modal!")
                exit()

        ###############################################################
        #
        #TEST
        #
        ###############################################################
        if self.config.is_test:
            test_modal = self.config.test_changed_modal
            test_pct = self.config.test_changed_pct
            if test_modal == 'language':
                utterance = utterance_text
            elif test_modal == 'video':
                utterance = utterance_video
            elif test_modal == 'audio':
                utterance = utterance_audio
            else:
                print("Wrong test_modal!")
                exit()
            if self.config.test_method == 'missing':
                for i, _ in enumerate(utterance):
                    rand_num = torch.rand(1)
                    if rand_num < test_pct:
                        utterance[i] = utterance[i] * 0
            elif self.config.test_method == 'g_noise':
                noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float())
                sample_num = int(len(utterance) * test_pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * noise[i]
            else:
                print("Wrong test method!")
                exit()

            if test_modal == 'language':
                utterance_text = utterance
            elif test_modal == 'video':
                utterance_video = utterance
            elif test_modal == 'audio':
                utterance_audio = utterance
            else:
                print("Wrong test_modal!")
                exit()


        # print("====== language multiply missing ======")
        # utterance_text = utterance_text * 0

        # print("----language multiply missing 30%-------")
        # for i, _ in enumerate(utterance_text):
        #     rand_num = torch.rand(1)
        #     if rand_num < 0.3:
        #         utterance_text[i] = utterance_text[i] * 0

        # print("--------------language multiply noise-------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance_text.size()[1])).float())
        # utterance_text = utterance_text * noise

        # print("-------------language multiply noise  30%---------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance_text.size()[0])).float())
        # sample_num = int(len(utterance_text) * 0.3)
        # sample_list = [i for i in range(len(utterance_text))]
        # sample_list = random.sample(sample_list, sample_num)
        # for i in sample_list:
        #     utterance_text[i] = utterance_text[i] * noise[i]
        ##################################################
        # print("====== video multiply missing ======")
        # utterance_video = utterance_video * 0

        # print("----video multiply missing 30%-------")
        # for i, _ in enumerate(utterance_video):
        #     rand_num = torch.rand(1)
        #     if rand_num < 0.3:
        #         utterance_video[i] = utterance_video[i] * 0

        # print("--------------video multiply noise-------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance_video.size()[1])).float())
        # utterance_video = utterance_video * noise

        # print("-------------video multiply noise  30%---------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance_video.size()[0])).float())
        # sample_num = int(len(utterance_video) * 0.3)
        # sample_list = [i for i in range(len(utterance_video))]
        # sample_list = random.sample(sample_list, sample_num)
        # for i in sample_list:
        #     utterance_video[i] = utterance_video[i] * noise[i]
        ##################################################
        # print("====== audio multiply missing ======")
        # utterance_audio = utterance_audio * 0

        # print("----audio multiply missing 30%-------")
        # for i, _ in enumerate(utterance_audio):
        #     rand_num = torch.rand(1)
        #     if rand_num < 0.3:
        #         utterance_audio[i] = utterance_audio[i] * 0

        # print("--------------audio multiply noise-------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance_audio.size()[1])).float())
        # utterance_audio = utterance_audio * noise

        # print("-------------utterance_audio multiply noise  30%---------------------")
        # noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance_audio.size()[0])).float())
        # sample_num = int(len(utterance_audio) * 0.3)
        # sample_list = [i for i in range(len(utterance_audio))]
        # sample_list = random.sample(sample_list, sample_num)
        # for i in sample_list:
        #     utterance_audio[i] = utterance_audio[i] * noise[i]


        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)


        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None


        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        o = self.fusion(h)
        return o
    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)


    def forward(self, is_train, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        batch_size = lengths.size(0)
        o = self.alignment(is_train, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o
