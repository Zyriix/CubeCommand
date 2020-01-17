import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import json
import random

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder,self).__init__()

        for key, val in args.__dict__.items():
            self.__setattr__(key, val)

        self.lstm = nn.LSTM(self.attitude_dim,
                            self.enc_hid_dim,
                            self.lstm_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        self.fc=nn.Linear(self.enc_hid_dim, self.dec_hid_dim)
        self.dropout=nn.Dropout(self.dropout)
    
    def forward(self, att_data):
        out, attitude_hidden = self.lstm(att_data)
        attitude_hidden = torch.tanh(self.fc(attitude_hidden[-1][-1]))
        return out, attitude_hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim,dec_hid_dim):
        super(Attention,self).__init__()
    
        self.enc_hid_dim=enc_hid_dim
        self.dec_hid_dim=dec_hid_dim

        self.attn=nn.Linear(enc_hid_dim+dec_hid_dim,dec_hid_dim)
        self.v=nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size=encoder_outputs.shape[1]
        src_len=encoder_outputs.shape[0]
        hidden=hidden.unsqueeze(1).repeat(1,src_len,1)
        encoder_outputs=encoder_outputs.permute(1,0,2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        energy = energy.permute(0, 2, 1)

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)

        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, args, attention):
        super(Decoder,self).__init__()
        self.attention = attention

        self.rnn=nn.LSTM(self.attitude_dim + self.enc_hid_dim,
                            self.dec_hid_dim,
                            self.lstm_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        self.out=nn.Linear(self.enc_hid_dim+self.dec_hid_dim+self.attitude_dim,self.output_dim)
        self.dropout=nn.Dropout(self.dropout)

    def forward(self,input,hidden,encoder_outputs):
        input=input.unsqueeze(0)

        a=self.attention(hidden,encoder_outputs)
        a=a.unsqueeze(1)
        encoder_outputs=encoder_outputs.permute(1,0,2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((input, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()

        input = input.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, input), dim = 1))

        return output, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder,decoder,device):
        super(Seq2Seq,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.device=device
    
    def forward(self, src,trg,teacher_forcing_ratio=0.5):
        batch_size=src.shape[1]
        max_len=trg.shape[0]
        trg_vocab_size=self.decoder.output_dim
        
        outputs=torch.zeros(max_len,batch_size,trg_vocab_size).to(self.device)
        # torch.Size([21, 128, 5893])
        encoder_outputs,hidden=self.encoder(src)
        
        output=trg[0,:]
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs