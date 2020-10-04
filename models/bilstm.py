# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
import numpy as np

class ClauseCNN(nn.Module):
    def __init__(self, opt):
        super(ClauseCNN, self).__init__()
        self.opt = opt
        self.conv11 = nn.Conv2d(1, opt.hidden_dim // 4, (2, opt.embed_dim)) 
        self.conv12 = nn.Conv2d(1, opt.hidden_dim // 4, (3, opt.embed_dim)) 
        self.conv13 = nn.Conv2d(1, opt.hidden_dim // 4, (4, opt.embed_dim)) 
        self.conv14 = nn.Conv2d(1, opt.hidden_dim // 4, (5, opt.embed_dim)) 
        self.dropout = nn.Dropout(opt.dropout) 

    @staticmethod 
    def conv_and_pool(x, conv): 
        x = conv(x) 
        x = F.relu(x.squeeze(3)) 
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x1 = self.conv_and_pool(x, self.conv11) 
        x2 = self.conv_and_pool(x, self.conv12) 
        x3 = self.conv_and_pool(x, self.conv13)
        x4 = self.conv_and_pool(x, self.conv14)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.dropout(x)
        return x

'''
usage:
self.pair_biaffine = Biaffine(opt, opt.hidden_dim, opt.hidden_dim, 1, bias=(True, False))
opt.hidden_dim == 300
'''

class Biaffine(nn.Module):
    def __init__(self, opt, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.opt = opt
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.opt.device)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.opt.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine

class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(BiLSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.dropout = nn.Dropout(0.5)
        self.clause_cnn = ClauseCNN(opt)
        self.bilstm = DynamicLSTM(opt.hidden_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.emotion_fc = nn.Linear(2 * opt.hidden_dim, 2 * opt.MLP_out_dim)
        self.cause_fc = nn.Linear(2 * opt.hidden_dim, 2 * opt.MLP_out_dim)
        self.pair_biaffine = Biaffine(opt, opt.MLP_out_dim, opt.MLP_out_dim, 1, bias=(True, False))
        self.emotion_fc1 = nn.Linear(opt.MLP_out_dim, opt.polarities_dim)
        self.cause_fc1 = nn.Linear(opt.MLP_out_dim, opt.polarities_dim)

    def calc_loss(self, outputs, targets):
        output_emotion, output_cause, output_pair = outputs
        target_emotion, target_cause, target_pair = targets
        emotion_loss = F.cross_entropy(output_emotion.flatten(0, 1), target_emotion.flatten(0, 1))
        cause_loss = F.cross_entropy(output_cause.flatten(0, 1), target_cause.flatten(0, 1))
        pair_loss = F.binary_cross_entropy(output_pair.flatten(1, 2), target_pair.flatten(1, 2))
        return emotion_loss + cause_loss + pair_loss
    

    def forward(self, inputs):
        doc_len, text_indices, pos_matr = inputs
        text = self.dropout(self.embed(text_indices)) 
        clause_chunks = torch.chunk(text, text.size(1), dim=1)
        clause_reps = []
        for clause_chunk in clause_chunks:
            clause_reps.append(self.clause_cnn(clause_chunk).unsqueeze(1))
        clause_rep = torch.cat(clause_reps, dim=1) 
        out, (_, _) = self.bilstm(clause_rep, doc_len) 
        emotion_rep = F.relu(self.emotion_fc(out))
        cause_rep = F.relu(self.cause_fc(out))
        emotion_node, emotion_rep = torch.chunk(emotion_rep, 2, dim=2)
        cause_node, cause_rep = torch.chunk(cause_rep, 2, dim=2)
        pair_out = self.pair_biaffine(emotion_node, cause_node).squeeze(3)
        emotion_out = self.emotion_fc1(emotion_rep)
        cause_out = self.cause_fc1(cause_rep)
        pair_out = torch.sigmoid(pair_out) * pos_matr
        
        return emotion_out, cause_out, pair_out