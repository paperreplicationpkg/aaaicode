#!/usr/bin python3
# -*- coding: utf-8 -*-
# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class baseline_lstm(nn.Module):
    def __init__(self, n_features):
        super(baseline_lstm, self).__init__()

        self.rnn = nn.LSTM(input_size=n_features, hidden_size=64, num_layers=2, batch_first=True)

        self.hidden1 = torch.nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.hidden2 = torch.nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(p=0.5)

        self.out = nn.Linear(32, 2)  # 输出层

        # 初始化层
        nn.init.xavier_normal_(self.rnn.all_weights[0][0])
        nn.init.xavier_normal_(self.rnn.all_weights[0][1])
        nn.init.xavier_normal_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.1)
        nn.init.xavier_normal_(self.hidden1.weight)
        nn.init.constant_(self.hidden1.bias, 0.1)
        nn.init.xavier_normal_(self.hidden2.weight)
        nn.init.constant_(self.hidden2.bias, 0.1)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = torch.cat((h_n[-1], h_n[-2]), dim=1)
        print(out.size())
        out = F.relu(self.bn1(self.hidden1(out)))
        out = F.relu(self.bn2(self.hidden2(out)))
        out = F.relu(self.dropout(out))
        out = self.out(out)

        return out


if __name__ == "__main__":
    model = baseline_lstm(65)
    print(model)