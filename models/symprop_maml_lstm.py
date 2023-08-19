#!/usr/bin python3
# -*- coding: utf-8 -*-
# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.vars = nn.ParameterList()  ## 包含了所有需要被优化的tensor
        self.vars_bn = nn.ParameterList()

        # self.hidden_size = 128
        # self.input_size = n_features

        # 第一个rnn层
        # W_ih = nn.Parameter(torch.ones((4 * self.hidden_size, self.input_size)))
        # W_hh = nn.Parameter(torch.ones((4 * self.hidden_size, self.hidden_size)))
        # b_ih = nn.Parameter(torch.ones((4 * self.hidden_size)))
        # b_hh = nn.Parameter(torch.ones((4 * self.hidden_size)))
        # nn.init.kaiming_normal_(W_ih)
        # nn.init.kaiming_normal_(W_hh)
        # self.vars.extend([W_ih, W_hh, b_ih, b_hh])

        # 第1个linear层
        # nn.linear(130000, 256)
        weight = nn.Parameter(torch.ones([256, 13000]))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])

        # 第1个BatchNorm层
        weight = nn.Parameter(torch.ones(256))
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])

        running_mean = nn.Parameter(torch.zeros(256), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(256), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # 第2个linear层
        # nn.linear(64, 32)
        weight = nn.Parameter(torch.ones([32, 256]))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(32))
        self.vars.extend([weight, bias])

        # 第2个BatchNorm层
        weight = nn.Parameter(torch.ones(32))
        bias = nn.Parameter(torch.zeros(32))
        self.vars.extend([weight, bias])

        running_mean = nn.Parameter(torch.zeros(32), requires_grad=False)
        running_var = nn.Parameter(torch.zeros(32), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])

        # 输出层
        # nn.linear(32, 2)
        weight = nn.Parameter(torch.ones([2, 32]))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(2))
        self.vars.extend([weight, bias])

    def forward(self, x, params=None, bn_training=True):

        if params is None:
            params = self.vars
            
        # print("before permute: ", x.size())
        # x = x.permute(1, 0, 2)
        # print("after permute: ", x.size())

        # W_ih, W_hh, b_ih, b_hh = params[0], params[1], params[2], params[3]
        # _, h1, _ = self.lstm_forward(x, W_ih, W_hh, b_ih, b_hh)
        # W_ih, W_hh, b_ih, b_hh = params[4], params[5], params[6], params[7]
        # _, h2, _ = self.lstm_forward(x, W_ih, W_hh, b_ih, b_hh)
        # print(h1.size())
        # x = torch.cat((h1, h2), dim=1)
        # print(x.size())
        # x = h1

        x = x.reshape(-1, 13000)

        # 第1个linear层
        weight, bias = params[0], params[1]
        x = F.relu(F.linear(x, weight, bias))

        # 第1个BN层
        weight, bias = params[2], params[3]
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)

        # 第2个linear层
        weight, bias = params[4], params[5]
        x = F.relu(F.linear(x, weight, bias))

        # 第2个BN层
        weight, bias = params[6], params[7]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=weight, bias=bias, training=bn_training)

        # 输出层
        weight, bias = params[8], params[9]
        x = F.linear(x, weight, bias)

        # output = F.softmax(x)
        output = x

        return output

    def parameters(self):

        return self.vars

    def lstm_forward(self, x, W_ih, W_hh, b_ih, b_hh, h=None, c=None):
        """
        x.shape: [time_step_length, batch_size, feature_dim]
        """
        batch_size = x.shape[1]
        if not h and not c:
            h, c = torch.zeros((batch_size, self.hidden_size)).cuda(), torch.zeros((batch_size, self.hidden_size)).cuda()
        outputs = []  # 存放隐藏状态
        for t in x:  # 遍历时间步
            # gates_shape: (n * (h*4)) 二维；即四个中间状态（3门+1个cell state）被连成一个一维向量，一共有n个这样的向量
            gates = torch.matmul(t, W_ih.t()) + b_ih + torch.matmul(h, W_hh.t()) + b_hh  # 生成未激活的中间状态
            i = torch.sigmoid(gates[:, 0 : self.hidden_size])  # input_gate
            f = torch.sigmoid(gates[:, self.hidden_size : 2 * self.hidden_size])  # forget_gate
            c_tilda = torch.tanh(gates[:, 2 * self.hidden_size : 3 * self.hidden_size])  # 候选细胞状态
            o = torch.sigmoid(gates[:, 3 * self.hidden_size :])  # output_gate
            c = f * c + i * c_tilda  # 更新cell state
            h = o * torch.tanh(c)  # 更新hidden state
            outputs.append(h)  # 将隐藏状态存入数组

        outputs = torch.stack(outputs)  # (T, B, h)或（T，B，2*h)
        return outputs, h, c  # 返回output，和时刻t的hidden state和cell state


if __name__ == "__main__":
    net = BaseNet(n_features=65)
