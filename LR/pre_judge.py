
import torch
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能 
import torch.optim as optim       # 模型优化器模块

from torch.utils.data.dataloader import DataLoader
torch.manual_seed(1)
print("done")
import numpy as np

import argparse
import time

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='LSTM CUBE classification')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training [default: 1]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_false',
                    help='enables cuda')

parser.add_argument('--save', type=str, default='./model/PRE_JUDGE.pt',
                    help='path to save the final model')


parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
# parser.add_argument('--embed-dim', type=int, default=64,
#                     help='number of embedding dimension [default: 64]')
# parser.add_argument('--rotate-dim', type=int, default=2,
#                     help='number of rotate data dimension [default:2 ]')
parser.add_argument('--input-dim', type=int, default=4,
                    help='number of input data dimension [default:4 ]')
parser.add_argument('--hidden-dim', type=int, default=512,
                    help='number of input data dimension [default:64 ]')
parser.add_argument('--label-num', type=int, default=2,
                    help='label numbers')
parser.add_argument('--data', type=str, default='./data/train_01.json',
                    help='location of the data corpus')
parser.add_argument('--test-data', type=str, default='./data/test_01.json',
                    help='location of the data corpus')

args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and args.cuda_able
print(use_cuda)
# ##############################################################################
# Load data
###############################################################################
from  dataset import PreDataset 

root = "d:/mcube/python/LR"
#魔方24种转动方法
train_data=PreDataset(datapath=args.data,t = "train")
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

test_data=PreDataset(datapath=args.test_data, t = "test")
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)



# ##############################################################################
# Build model
# ##############################################################################
from model import LR

lr = LR(args)
if use_cuda:
    lr = lr.cuda()

optimizer = torch.optim.SGD(lr.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

# ##############################################################################
# Training
# ##############################################################################
import time
from tqdm import tqdm

train_loss = []
valid_loss = []
accuracy = []

def evaluate():
    lr.eval()
    corrects = eval_loss = 0
    _size = test_loader.__len__()
    
    for data, label in tqdm(test_loader, mininterval=0.2,
                desc='Test Processing', leave=False):
        if args.cuda_able:
            data=data.cuda()
            label = label.cuda()
        target = lr(data.float())
        loss = criterion(target, label)
        eval_loss += loss.data
        pre = torch.max(target, 1)[1]

        corrects += [0,1][int(pre) == int(label.data)]
    return eval_loss/_size, corrects, float(corrects)/_size * 100.0, _size
        

def train():
    lr.train()
    total_loss = 0
    # att_hidden = lstm.init_hidden()
    # rotate_hidden = lstm.init_hidden()

##debug
    for data, label in tqdm(train_loader, mininterval=1,
                desc='Train Processing', leave=False): 
        if args.cuda_able:
            data=data.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        target = lr(data.float())
        # print(target,label)
        loss = criterion(target, label)
        loss.backward()
        # print(loss.data)
        optimizer.step()
        total_loss+=loss.data
    return total_loss/train_loader.__len__()

##origin
    # for att_data,rotate_data, label in tqdm(train_loader, mininterval=1,
    #             desc='Train Processing', leave=False):
    #     optimizer.zero_grad()
    #     # hidden = repackage_hidden(hidden)
    #     # print(att_hidden.size())
    #     # print(att_data.size())
    #     # print(rotate_data.size())
    #     target, att_hidden, rotate_hidden = lstm(att_data.float() , rotate_data.float() , att_hidden , rotate_hidden )
    #     # print(target,label)
    #     target = target.view(1,24)
    #     loss = criterion(target, label)
    #     # loss = loss_tmp
    #     loss.backward(retain_graph=True)
    #     optimizer.step()
    #     # total_loss.append(loss.data) 
    #     print(loss.data)
    #     total_loss+=loss.data
    #     # print("loss:",sum(total_loss)/len(total_loss))
    # return total_loss/train_loader.__len__()

# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss*1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))
        # lstm.load_state_dict(torch.load(args.save))
        

        epoch_start_time = time.time()
        loss, corrects, acc, size = evaluate()
        valid_loss.append(loss*1000.)
        accuracy.append(acc)
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, loss, acc, corrects, size))
        print('-' * 90)
        if not best_acc or best_acc < corrects:
            best_acc = corrects
            model_state_dict = lr.state_dict()
            model_source = {
                "settings": args,
                "model": model_state_dict
                # "src_dict": data['dict']['train']
            }
            torch.save(model_state_dict, args.save)
except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

