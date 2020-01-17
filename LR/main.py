# Author: Robert Guthrie
# 作者：Robert Guthrie

import torch
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能 
import torch.optim as optim       # 模型优化器模块

from torch.utils.data.dataloader import DataLoader
torch.manual_seed(1)
import numpy as np
import argparse
import time
import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='LSTM CUBE classification')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=256,
                    help='batch size for training [default: 1]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_false',
                    help='enables cuda')
parser.add_argument('--save', type=str, default='./model/LSTM_CUBE.pt',
                    help='path to save the final model')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--input-dim', type=int, default=4,
                    help='number of input data dimension [default:4 ]')
parser.add_argument('--hidden-dim', type=int, default=1024,
                    help='number of  hidden dimension [default: 128]')
parser.add_argument('--label-num', type=int, default=24,
                    help='label numbers')
parser.add_argument('--data', type=str, default='./data/train.json',
                    help='location of the data corpus')
parser.add_argument('--test-data', type=str, default='./data/test.json',
                    help='location of the data corpus')
args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available() and args.cuda_able
print(use_cuda)
# ##############################################################################
# Load data
###############################################################################

from  dataset import PosDataset
root = "d:/mcube/python/LR"
#魔方24种转动方法
train_data=PosDataset(datapath=args.data,t = "train")
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

test_data=PosDataset(datapath=args.test_data, t = "test")
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)



# ##############################################################################
# Build model
# ##############################################################################
from model import LR

lr = LR(args)
if use_cuda:
    lr = lr.cuda()

optimizer = torch.optim.Adam(lr.parameters(), lr=args.lr, weight_decay=0.005)
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
    _size = test_data.__len__()
    
    for data, label in tqdm(test_loader, mininterval=0.2,
                desc='Test Processing', leave=False):
        if args.cuda_able:
            data=data.cuda()
            label = label.cuda()
        target = lr(data.float())
        loss = criterion(target, label)
        eval_loss += loss.data
        # print(pre)

        corrects += (torch.max(target, 1)[1].view(label.size()).data == label.data).sum()
    return eval_loss/test_loader.__len__(), corrects, float(corrects)/_size * 100.0, _size

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.3 ** (epoch // 10))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr       
   
    

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
        # print("loss:",float(loss.data))
        total_loss+=loss.data
    return total_loss/train_loader.__len__()


   

# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=1, patience=3)
try:
    print('-' * 90)
    for epoch in range(1, args.epochs+1):
        adjust_learning_rate(optimizer,epoch)
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss*1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))
        # lstm.load_state_dict(torch.load(args.save))
        

        epoch_start_time = time.time()
        loss, corrects, acc, size = evaluate()
        scheduler.step(acc)
        valid_loss.append(loss*1000.)
        accuracy.append(acc)
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | valiate loss {:.4f} | accuracy {:.4f}%({}/{})'.format(epoch, time.time() - epoch_start_time, loss, acc, corrects, size))
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

