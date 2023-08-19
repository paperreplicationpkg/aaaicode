#!/usr/bin python3
# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F

class fcn_net(nn.Module):
    def __init__(self, n_in, n_classes, n_features):
        super(fcn_net, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes
        self.n_features = n_features
        
        self.conv1 = nn.Conv2d(1, 128, (8, 1), 1, (3, 0))
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, (5, 1), 1, (2, 0))
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 128, (3, 1), 1, (1, 0))
        self.bn3 = nn.BatchNorm2d(128)

        self.fc4 = nn.Linear(128, self.n_classes)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1, self.n_in, self.n_features)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, 4)
        x = torch.mean(x, dim=2)
        x = torch.mean(x, dim=2)
        x = x.view(-1, 128)
        x = self.fc4(x)
        x = F.log_softmax(x, 1)
        return x