# -*- coding: utf-8 -*-

import os
import pickle
import random
import numpy as np

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './w2v_200.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        found = 0
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                found += 1
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
        print('percentage of words found in pretrained word vectors {}/{}'.format(found, len(word2idx)))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ECPEDataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ECPEDatesetReader:
    @staticmethod
    def __read_text__(fnames):
        all_text = ''
        max_doc_len = 0
        max_seq_len = 0
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            while True:
                line = fin.readline()
                if line == '': 
                    break
                line = line.strip().split()
                doc_len = int(line[1])
                if doc_len > max_doc_len:
                    max_doc_len = doc_len
                pairs = eval('[' + fin.readline().strip() + ']')
                for i in range(doc_len):
                    text = fin.readline().strip().split(',')[-1]
                    if len(text.split()) > max_seq_len:
                        max_seq_len = len(text.split())
                    all_text += (text + ' ')
            fin.close()
        return all_text, max_doc_len, max_seq_len

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        
        all_data = []
        pos_matr_all = []
        while True:
            line = fin.readline()
            if line == '': break
            line = line.strip().split()
            doc_len = int(line[1])
            pairs = eval('[' + fin.readline().strip() + ']')
            y_pair = np.zeros((doc_len, doc_len), dtype=np.float32)
            for pair in pairs:
                y_pair[pair[0]-1, pair[1]-1] = 1
            emotion_indicator, cause_indicator = zip(*pairs)
            y_emotion = []
            y_cause = []
            text_indices = []
            seq_lens = []
            pos_matr = np.zeros((doc_len,doc_len),dtype='float32')
            for i in range(doc_len):
                if (i+1) in emotion_indicator:
                    y_emotion.append(1)
                else:
                    y_emotion.append(0)
                if (i+1) in cause_indicator:
                    y_cause.append(1)
                else:
                    y_cause.append(0)
                clause = fin.readline().strip().split(',')[-1]
                seq_lens.append(len(clause.split()))
                clause_indices = tokenizer.text_to_sequence(clause)
                text_indices.append(clause_indices)
            for i in range(doc_len):
                for j in range(doc_len):
                    pos_matr[i][j] = (doc_len - abs(i - j - 1) + 1) / (doc_len + 1.0)
            pos_matr_all.append(pos_matr)
            assert len(y_emotion) == len(y_cause) == len(text_indices) == len(seq_lens) == doc_len
            data = {
                'seq_lens': seq_lens,
                'text_indices': text_indices,
                'y_emotion': y_emotion,
                'y_cause': y_cause,
                'y_pair': y_pair,
                'pos_matr' : pos_matr,
            }
            all_data.append(data)

        fin.close()
        return all_data

    def __init__(self, dataset='sina', embed_dim=300):
        print("preparing {0} dataset...".format(dataset))
        fname = {
            'sina': './datasets/sina/',
        }
        fnames = [fname[dataset]+f'fold{i+1}_train.txt' for i in range(10)] + [fname[dataset]+f'fold{i+1}_test.txt' for i in range(10)]
        text, self.max_doc_len, self.max_seq_len = ECPEDatesetReader.__read_text__(fnames)
        
        if os.path.exists(dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer.word2idx, f)
        
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = [ECPEDataset(ECPEDatesetReader.__read_data__(fname[dataset]+f'fold{i+1}_train.txt', tokenizer)) for i in range(10)]
        self.test_data = [ECPEDataset(ECPEDatesetReader.__read_data__(fname[dataset]+f'fold{i+1}_test.txt', tokenizer)) for i in range(10)]
