import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import json


class LR(nn.Module,):
  def __init__(self, args):
    super(LR, self).__init__()
    for k, v in args.__dict__.items():
        self.__setattr__(k,v)
    self.lr = nn.Sequential(
      nn.Linear(self.input_dim,self.hidden_dim),
      nn.BatchNorm1d(self.hidden_dim),
      nn.ReLU(),
      nn.Linear(self.hidden_dim,int(self.hidden_dim/2)),
      nn.BatchNorm1d(int(self.hidden_dim/2)),
      nn.ReLU(),
      
      nn.Linear(int(self.hidden_dim/2), self.label_num),
      nn.BatchNorm1d( self.label_num),
      nn.ReLU(),
      nn.LogSoftmax()
    )
  def _init_weights(self, scope=1.):
        self.fc1.weight.data.uniform_(-scope, scope)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-scope, scope)
        self.fc2.bias.data.fill_(0)
  def forward(self, input):
    out = self.lr(input)
    return out

