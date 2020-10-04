# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ECPEDatesetReader
from models import BiLSTM
from matplotlib import pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class Instructor:
    def __init__(self, opt):
        self.opt = opt

        self.dataset = ECPEDatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        self.model = opt.model_class(self.dataset.embedding_matrix, opt).to(opt.device)
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('>> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('>> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, optimizer, fold_i):
        max_test_precision = 0
        max_test_recall = 0
        max_test_f1 = 0

        max_test_precision_e = 0
        max_test_recall_e = 0
        max_test_f1_e = 0
        max_test_precision_c = 0
        max_test_recall_c = 0
        max_test_f1_c = 0
        max_pair_matrix_len = 20
        global_step = 0
        continue_not_increase = 0
        train_data_loader = BucketIterator(data=self.dataset.train_data[fold_i], batch_size=opt.batch_size, shuffle=True)
        test_data_loader = BucketIterator(data=self.dataset.test_data[fold_i], batch_size=opt.batch_size, shuffle=False)
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            increase_flag = False
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = [sample_batched[col].to(self.opt.device) for col in self.opt.targets_cols]

                outputs = self.model(inputs)
                loss = self.model.calc_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                if global_step % self.opt.log_step == 0:
                    output_emotion, output_cause, output_pair = outputs
                    target_emotion, target_cause, target_pair = targets
                    output_pairs = torch.nonzero(output_pair > 0.3).cpu().numpy().tolist()
                    target_pairs = torch.nonzero(target_pair > 0.3).cpu().numpy().tolist()
                    n_TP = 0
                    for i_pair in output_pairs:
                        if i_pair in target_pairs:
                            n_TP += 1
                    n_FP = (len(output_pairs) - n_TP)
                    n_FN = (len(target_pairs) - n_TP)
                    precision = float(n_TP) / float(n_TP + n_FP + 1e-5)
                    recall = float(n_TP) / float(n_TP + n_FN + 1e-5)
                    f1 = 2 * precision * recall / (precision + recall + 1e-5)

                    test_precision, test_recall, test_f1, test_precision_e, test_recall_e, test_f1_e, test_precision_c, test_recall_c, test_f1_c = self._evaluate(test_data_loader)
                    
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_precision = test_precision
                        max_test_recall = test_recall 
                        max_test_f1 = test_f1
                        if self.opt.save:
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            torch.save(self.model.state_dict(), 'state_dict/'+self.opt.model_name+'_' + str(fold_i) +self.opt.dataset+'.pkl')
                            print('>> model saved.')
                    print('loss: {:.4f}, train_f1: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}'.format(loss.item(), f1, test_precision, test_recall, test_f1))
                    if test_f1_e > max_test_f1_e:
                        increase_flag = True
                        max_test_precision_e = test_precision_e
                        max_test_recall_e = test_recall_e 
                        max_test_f1_e = test_f1_e   
                    print('test_precision_e: {:.4f}, test_recall_e: {:.4f}, test_f1_e: {:.4f}'.format(test_precision_e, test_recall_e, test_f1_e))
                    if test_f1_c > max_test_f1_c:
                        increase_flag = True
                        max_test_precision_c = test_precision_c
                        max_test_recall_c = test_recall_c 
                        max_test_f1_c = test_f1_c
                    print('test_precision_c: {:.4f}, test_recall_c: {:.4f}, test_f1_c: {:.4f}'.format(test_precision_c, test_recall_c, test_f1_c))
                    
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= self.opt.patience:
                    print('>> early stop.')
                    break
            else:
                continue_not_increase = 0              
        return max_test_precision, max_test_recall, max_test_f1, max_test_precision_e, max_test_recall_e, max_test_f1_e, max_test_precision_c, max_test_recall_c, max_test_f1_c

    def _evaluate(self, test_data_loader):
        # switch model to evaluation mode
        self.model.eval()
        t_n_TP, t_n_FP, t_n_FN = 0, 0, 0
        t_n_TP_e, t_n_FP_e, t_n_FN_e = 0, 0, 0
        t_n_TP_c, t_n_FP_c, t_n_FN_c = 0, 0, 0
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(test_data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = [t_sample_batched[col].to(self.opt.device) for col in self.opt.targets_cols]
                t_outputs = self.model(t_inputs)

                t_output_emotion, t_output_cause, t_output_pair = t_outputs
                t_target_emotion, t_target_cause, t_target_pair = t_targets
                t_output_pairs = torch.nonzero(t_output_pair > 0.3).cpu().numpy().tolist()
                t_target_pairs = torch.nonzero(t_target_pair > 0.3).cpu().numpy().tolist()

                t_output_emotions = torch.nonzero(torch.argmax(t_output_emotion, 2)).cpu().numpy().tolist()
                t_target_emotions = torch.nonzero(t_target_emotion).cpu().numpy().tolist()
                t_output_causes = torch.nonzero(torch.argmax(t_output_cause, 2)).cpu().numpy().tolist()
                t_target_causes = torch.nonzero(t_target_cause.cpu()).numpy().tolist()
                n_TP = 0
                n_TP_e = 0
                n_TP_c = 0
                for i_pair in t_output_pairs:
                    if i_pair in t_target_pairs:
                        n_TP += 1
                n_FP = (len(t_output_pairs) - n_TP)
                n_FN = (len(t_target_pairs) - n_TP)

                for i_emotion in t_output_emotions:
                    if i_emotion in t_target_emotions:
                        n_TP_e += 1
                for i_cause in t_output_causes:
                    if i_cause in t_target_causes:
                        n_TP_c += 1
                n_FP_e = (len(t_output_emotions) - n_TP_e)
                n_FN_e = (len(t_target_emotions) - n_TP_e)
                n_FP_c = (len(t_output_causes) - n_TP_c)
                n_FN_c = (len(t_target_causes) - n_TP_c)
                t_n_TP_e += n_TP_e
                t_n_FP_e += n_FP_e
                t_n_FN_e += n_FN_e
                t_n_TP_c += n_TP_c
                t_n_FP_c += n_FP_c
                t_n_FN_c += n_FN_c
                t_n_TP += n_TP
                t_n_FP += n_FP
                t_n_FN += n_FN

        precision = float(t_n_TP) / float(t_n_TP + t_n_FP + 1e-5)
        recall = float(t_n_TP) / float(t_n_TP + t_n_FN + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        precision_e = float(t_n_TP_e) / float(t_n_TP_e + t_n_FP_e + 1e-5)
        recall_e = float(t_n_TP_e) / float(t_n_TP_e + t_n_FN_e + 1e-5)
        f1_e = 2 * precision_e * recall_e / (precision_e + recall_e + 1e-5)
        precision_c = float(t_n_TP_c) / float(t_n_TP_c + t_n_FP_c + 1e-5)
        recall_c = float(t_n_TP_c) / float(t_n_TP_c + t_n_FN_c + 1e-5)
        f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + 1e-5)
        return precision, recall, f1, precision_e, recall_e, f1_e, precision_c, recall_c, f1_c

    def run(self, k_fold=10):
        all_f1 = []
        all_p = []
        all_r = []
        # Loss and Optimizer
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        max_test_precision_avg = 0
        max_test_recall_avg = 0
        max_test_f1_avg = 0
        max_test_precision_e_avg = 0
        max_test_recall_e_avg = 0
        max_test_f1_e_avg = 0
        max_test_precision_c_avg = 0
        max_test_recall_c_avg = 0
        max_test_f1_c_avg = 0

        for fold_i in range(k_fold):
            print('fold: ', fold_i+1)
            self._reset_params()
            max_test_precision, max_test_recall, max_test_f1, max_test_precision_e, max_test_recall_e, max_test_f1_e, max_test_precision_c, max_test_recall_c, max_test_f1_c = self._train(optimizer, fold_i)
            print('pair: max_test_precision: {:.4f}  max_test_recall: {:.4f}  max_test_f1: {:.4f}'.format(max_test_precision, max_test_recall, max_test_f1))
            print('emotion: max_test_precision: {:.4f}  max_test_recall: {:.4f}  max_test_f1: {:.4f}'.format(max_test_precision_e, max_test_recall_e, max_test_f1_e))
            print('cause: max_test_precision: {:.4f}  max_test_recall: {:.4f}  max_test_f1: {:.4f}'.format(max_test_precision_c, max_test_recall_c, max_test_f1_c))

            all_f1.append(max_test_f1)
            all_p.append(max_test_precision)
            all_r.append(max_test_recall)
            max_test_precision_avg += max_test_precision
            max_test_recall_avg += max_test_recall
            max_test_f1_avg += max_test_f1

            max_test_precision_e_avg += max_test_precision_e
            max_test_recall_e_avg += max_test_recall_e
            max_test_f1_e_avg += max_test_f1_e
            max_test_precision_c_avg += max_test_precision_c
            max_test_recall_c_avg += max_test_recall_c
            max_test_f1_c_avg += max_test_f1_c

            print('#' * 100)
        print("10-fold precision",all_p)
        print("10-fold recall",all_r)
        print("10-fold f1",all_f1)
        print("max_test_precision_avg:", max_test_precision_avg / k_fold)
        print("max_test_recall_avg:", max_test_recall_avg / k_fold)
        print("max_test_f1_avg:", max_test_f1_avg / k_fold)

        print("emotion:max_test_precision_avg:", max_test_precision_e_avg / k_fold)
        print("emotion:max_test_recall_avg:", max_test_recall_e_avg / k_fold)
        print("emotion:max_test_f1_avg:", max_test_f1_e_avg / k_fold)
        print("cause:max_test_precision_avg:", max_test_precision_c_avg / k_fold)
        print("cause:max_test_recall_avg:", max_test_recall_c_avg / k_fold)
        print("cause:max_test_f1_avg:", max_test_f1_c_avg / k_fold)


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bilstm', type=str)
    parser.add_argument('--dataset', default='sina', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=200, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--MLP_out_dim', default=100, type=int)
    parser.add_argument('--polarities_dim', default=2, type=int)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    opt = parser.parse_args()

    model_classes = {
        'bilstm': BiLSTM,
    }
    input_colses = {
        'bilstm': ['doc_len', 'text_indices', 'pos_matr'],
    }
    target_colses = {
        'bilstm': ['y_emotion', 'y_cause', 'y_pair'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.targets_cols = target_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run()
