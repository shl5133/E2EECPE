# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy as np

class BucketIterator(object):
    def __init__(self, data, batch_size, shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.batches, self.max_doc_len, self.num_batch = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        max_doc_len = 0
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x['seq_lens']))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            padded_data, batch_max_doc_len = self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size])
            batches.append(padded_data)
            if batch_max_doc_len > max_doc_len:
                max_doc_len = batch_max_doc_len
        return batches, max_doc_len, num_batch

    @staticmethod
    def pad_data(batch_data):
        batch_doc_len = []
        batch_text_indices = []
        batch_y_emotion = []
        batch_y_cause = []
        batch_y_pair = []
        batch_pos_matr = []
        max_doc_len = max([len(t['seq_lens']) for t in batch_data])
        max_seq_len = max([max(t['seq_lens']) for t in batch_data])
        for item in batch_data:
            seq_lens, text_indices, y_emotion, y_cause, y_pair, pos_matr = \
                item['seq_lens'], item['text_indices'], item['y_emotion'], item['y_cause'], item['y_pair'], item['pos_matr']
            doc_len = len(seq_lens)
            batch_doc_len.append(doc_len)
            padded_text_indices = []
            for i, clause_indices in enumerate(text_indices):
                clause_padding = [0] * (max_seq_len - seq_lens[i])
                clause_indices = clause_indices + clause_padding
                padded_text_indices.append(clause_indices)
            text_padding = [[0] * max_seq_len] * (max_doc_len - doc_len)
            padded_text_indices = padded_text_indices + text_padding
            batch_text_indices.append(padded_text_indices)
            y_emotion_padding = [0] * (max_doc_len - doc_len)
            y_cause_padding = [0] * (max_doc_len - doc_len)
            batch_y_emotion.append(y_emotion + y_emotion_padding)
            batch_y_cause.append(y_cause + y_cause_padding)
            batch_y_pair.append(np.pad(y_pair, \
                ((0, max_doc_len - doc_len), (0, max_doc_len - doc_len)), 'constant')) # default padding 0s
            batch_pos_matr.append(np.pad(pos_matr, \
                ((0, max_doc_len - doc_len), (0, max_doc_len - doc_len)), 'constant'))
        return { 
            'doc_len': torch.tensor(batch_doc_len),
            'text_indices': torch.tensor(batch_text_indices), 
            'y_emotion': torch.tensor(batch_y_emotion), 
            'y_cause': torch.tensor(batch_y_cause), 
            'y_pair': torch.tensor(batch_y_pair),
            'pos_matr' :torch.tensor(batch_pos_matr),
        },max_doc_len

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
