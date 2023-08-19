#!/usr/bin python3
# -*- coding: utf-8 -*-
# here put the import lib
import os
import sys
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(""))
from models import symprop_maml_basenet


class MetaLearner(nn.Module):
    def __init__(self):
        super(MetaLearner, self).__init__()
        self.update_step = 15  ## task-level inner update steps
        self.update_step_test = 15
        self.net = symprop_maml_basenet.BaseNet()
        self.meta_lr = 0.001
        self.base_lr = 0.01
        #         self.meta_lr = 2e-4
        #         self.base_lr = 4 * 1e-2
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
        # self.rnn = nn.LSTM(input_size=n_features, hidden_size=128, num_layers=2, batch_first=True)
        # nn.init.xavier_normal_(self.rnn.all_weights)

    def forward(self, x_spt, y_spt, x_qry, y_qry):

        # 初始化
        # batch_size = task_num
        task_num, support_size, time_step, feature_num = x_spt.size()
        query_size = x_qry.size(1)
        loss_list_qry = [0 for _ in range(self.update_step + 1)]
        correct_list = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            y_hat = self.net(x_spt[i], params=None, bn_training=True)  # (ways * shots, ways)
            loss = F.cross_entropy(y_hat, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            tuples = zip(grad, self.net.parameters())  ## 将梯度和参数\theta一一对应起来
            # fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L) 更新basenet的参数
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

            # 在第一次更新前update
            # 在query集上测试，计算准确率
            # 使用self.net.parameters()进行前向传播
            with torch.no_grad():

                y_hat = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[0] += loss_qry
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                correct_list[0] += correct

            # 使用更新过一次的fast_weights在query集上测试
            with torch.no_grad():
                y_hat = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[1] += loss_qry
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
                correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                correct_list[1] += correct

            for k in range(1, self.update_step):
                y_hat = self.net(x_spt[i], params=fast_weights, bn_training=True)
                # print('y_hat: ', y_hat.shape, y_hat)
                loss = F.cross_entropy(y_hat, y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                tuples = zip(grad, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))

                y_hat = self.net(x_qry[i], params=fast_weights, bn_training=True)
                loss_qry = F.cross_entropy(y_hat, y_qry[i])
                loss_list_qry[k + 1] += loss_qry

                with torch.no_grad():
                    pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_qry, y_qry[i]).sum().item()
                    correct_list[k + 1] += correct

        loss_qry = loss_list_qry[-1] / task_num
        self.meta_optim.zero_grad()  # 梯度清零
        loss_qry.backward()
        self.meta_optim.step()

        accs = np.array(correct_list) / (query_size * task_num)
        loss_list_qry = [loss.detach().cpu().numpy() for loss in loss_list_qry]
        #         print(loss_list_qry)
        loss = np.array(loss_list_qry) / (task_num)
        return accs, loss

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        #         assert len(x_spt.shape) == 4

        query_size = x_qry.size(0)
        correct_list = [0 for _ in range(self.update_step_test + 1)]

        new_net = copy.deepcopy(self.net)
        y_hat = new_net(x_spt)
        loss = F.cross_entropy(y_hat, y_spt)
        grad = torch.autograd.grad(loss, new_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

        # 在query集上测试，计算准确率
        # 这一步使用更新前的数据
        with torch.no_grad():
            y_hat = new_net(x_qry, params=new_net.parameters(), bn_training=True)
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(pred_qry, y_qry).sum().item()
            correct_list[0] += correct

        # 使用更新后的数据在query集上测试。
        with torch.no_grad():

            y_hat = new_net(x_qry, params=fast_weights, bn_training=True)
            pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(pred_qry, y_qry).sum().item()
            correct_list[1] += correct

        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, params=fast_weights, bn_training=True)
            loss = F.cross_entropy(y_hat, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, fast_weights)))

            y_hat = new_net(x_qry, fast_weights, bn_training=True)

            with torch.no_grad():
                pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
                correct = torch.eq(pred_qry, y_qry).sum().item()
                correct_list[k + 1] += correct

        del new_net
        accs = np.array(correct_list) / query_size
        return accs

    def predict(self, x_spt, y_spt, predict_x):
        # query_size = x_qry.size(0)
        # correct_list = [0 for _ in range(self.update_step_test + 1)]

        new_net = copy.deepcopy(self.net)
        new_net.train()
        y_hat = new_net(x_spt)
        loss = F.cross_entropy(y_hat, y_spt)
        grad = torch.autograd.grad(loss, new_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

        # 在query集上测试，计算准确率
        # 这一步使用更新前的数据
        # with torch.no_grad():
        #     y_hat = new_net(x_qry, params=new_net.parameters(), bn_training=True)
        #     pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
        #     correct = torch.eq(pred_qry, y_qry).sum().item()
        #     correct_list[0] += correct

        # 使用更新后的数据在query集上测试。
        # with torch.no_grad():
        #     y_hat = new_net(x_qry, params=fast_weights, bn_training=True)
            # pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            # correct = torch.eq(pred_qry, y_qry).sum().item()
            # correct_list[1] += correct

        for k in range(1, self.update_step_test):
            y_hat = new_net(x_spt, params=fast_weights, bn_training=True)
            loss = F.cross_entropy(y_hat, y_spt)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, fast_weights)))

            # y_hat = new_net(x_qry, fast_weights, bn_training=True)

            # with torch.no_grad():
            #     pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
            #     correct = torch.eq(pred_qry, y_qry).sum().item()
                # correct_list[k + 1] += correct

        new_net.eval()
        pred_y = new_net(predict_x, fast_weights, bn_training=False)
        pred_y = F.softmax(pred_y, dim=1).argmax(dim=1)

        del new_net
        # accs = np.array(correct_list) / query_size
        # return accs, pred_y
        return pred_y

#     def finetuning_2_models(self, x_spt, y_spt, x_qry, y_qry, save_path):
#         query_size = x_qry.size(0)
#         correct_list = [0 for _ in range(self.update_step_test + 1)]

#         new_net = copy.deepcopy(self.net)
#         y_hat = new_net(x_spt)
#         loss = F.cross_entropy(y_hat, y_spt)
#         grad = torch.autograd.grad(loss, new_net.parameters())
#         fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, new_net.parameters())))

#         # 在query集上测试，计算准确率
#         # 这一步使用更新前的数据
#         with torch.no_grad():
#             y_hat = new_net(x_qry, params=new_net.parameters(), bn_training=True)
#             pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
#             correct = torch.eq(pred_qry, y_qry).sum().item()
#             correct_list[0] += correct

#         # 使用更新后的数据在query集上测试。
#         with torch.no_grad():

#             y_hat = new_net(x_qry, params=fast_weights, bn_training=True)
#             pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
#             correct = torch.eq(pred_qry, y_qry).sum().item()
#             correct_list[1] += correct

#         for k in range(1, self.update_step_test):
#             y_hat = new_net(x_spt, params=fast_weights, bn_training=True)
#             loss = F.cross_entropy(y_hat, y_spt)
#             grad = torch.autograd.grad(loss, fast_weights)
#             fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], zip(grad, fast_weights)))

#             y_hat = new_net(x_qry, fast_weights, bn_training=True)

#             with torch.no_grad():
#                 pred_qry = F.softmax(y_hat, dim=1).argmax(dim=1)
#                 correct = torch.eq(pred_qry, y_qry).sum().item()
#                 correct_list[k + 1] += correct

#         torch.save(new_net.state_dict(), save_path)
#         del new_net
#         accs = np.array(correct_list) / query_size
#         return accs