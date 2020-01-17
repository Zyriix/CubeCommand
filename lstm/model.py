import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import json

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)

class AttnDecoderRNN(nn.Module):
    def __init__(self, args):
        super(AttnDecoderRNN, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.hidden_size = self.hidden_size
        self.output_size = self.output_size
        self.dropout_p = self.dropout_p
        self.max_length = self.max_length

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.attitude_dim,
                            self.hidden_size,
                            self.lstm_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1) # 连接输入的词向量和上一步的hide state并建立bp训练，他们决定了attention权重
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0)) # 施加权重到所有的语义向量上

        output = torch.cat((embedded[0], attn_applied[0]), 1) # 加了attention的语义向量和输入的词向量共同作为输入，此处对应解码方式三+attention
        output = self.attn_combine(output).unsqueeze(0) # 进入RNN之前，先过了一个全连接层

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1) # 输出分类结果
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

class LSTM_CUBE     (nn.Module):
    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.num_directions = 2 if self.bidirectional else 1

        # self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim,
        #                                  padding_idx=const.PAD)

        self.lstm_for_attitude = nn.LSTM(self.attitude_dim,
                            self.hidden_size,
                            self.lstm_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        # self.lstm_for_rotate = nn.LSTM(self.rotate_dim,
        #                     self.hidden_size,
        #                     self.lstm_layers,
        #                     dropout=self.dropout,
        #                     batch_first=True,
        #                     bidirectional=self.bidirectional)
        
        

        
        self.ln = LayerNorm(self.hidden_size * self.num_directions)

        self.logistic1 = nn.Linear(self.hidden_size * self.num_directions,
                                  256)
        # self.logistic2 = nn.Linear(512,256)
        self.logistic3 = nn.Linear(256,self.label_size)

        self._init_weights()
    

    def _init_weights(self, scope=1.):
        # self.lookup_table.weight.data.uniform_(-scope, scope)
        self.logistic1.weight.data.uniform_(-scope, scope)
        self.logistic1.bias.data.fill_(0)
        self.logistic2.weight.data.uniform_(-scope, scope)
        self.logistic2.bias.data.fill_(0)
        # self.logistic3.weight.data.uniform_(-scope, scope)
        # self.logistic3.bias.data.fill_(0)

    def init_hidden(self):
        num_layers = self.lstm_layers * self.num_directions

        weight = next(self.parameters()).data
        return (weight.new_zeros(num_layers, self.batch_size, self.hidden_size).float(), weight.new_zeros(num_layers, self.batch_size, self.hidden_size).float())

    # def forward(self,attitude_input, rotate_input, attitude_hidden, rotate_hidden):
        # encode = self.lookup_table(input)
        # print(attitude_hidden[0]).size(),attitude_hidden[1]

        # lstm_attitude_out, attitude_hidden = self.lstm_for_attitude(attitude_input, attitude_hidden)
        # lstm_rotate_out, rotate_hidden = self.lstm_for_rotate(rotate_input, rotate_hidden)

        # lstm_attitude_output = self.ln(lstm_attitude_out)[-1]
        # lstm_rotate_output =  self.ln(lstm_rotate_out)[-1]
        # print("att",lstm_attitude_output.size())
        # print("rotate",lstm_rotate_output.size())
        # output = torch.cat((lstm_attitude_output[-1],lstm_rotate_output[-1]),-1)
        # output = torch.cat((lstm_attitude_output[-1],lstm_rotate_output[-1]),-1)
        # print("out",output.size())
        # return F.log_softmax(self.logistic(output)), attitude_hidden, rotate_hidden

        ##debug
    def getFinalLabel(self,steps,att ):
        with open("d:/mcube/python/lstm/data/maps.txt",'r') as m:
          
            maps=json.load(m)
            
            # print(maps)

            for key,value in maps.items():
                # print([str(int(steps.data)),str(int(att.data))],value)
                if [steps,str(int(att.data))] in value:
                    # print(key, [steps,str(int(att.data))])
                    return key
        
        return 0

    def forward(self,attitude_input):
          out, attitude_hidden = self.lstm_for_attitude(attitude_input)
        #   out = self.ln(lstm_attitude_out)
          out = torch.tanh(self.logistic1(out))
        #   out = torch.tanh(self.logistic2(out))
          
          out = self.logistic3(out)[-1][-1]
          
        #   print(out.size())
        #   t = torch.mm(lstm_attitude_out[0][-1][:].view(1,-1),(torch.ones(128, 24)))
        #   print(out.size())
        #   print(t.size())
          return F.log_softmax(out.view(1,self.label_size))
