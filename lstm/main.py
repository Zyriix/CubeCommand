import argparse
import torch
parser = argparse.ArgumentParser(description='LSTM CUBE classification')
parser.add_argument('--lr', type=float, default=0.003,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training [default: 1]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_false',
                    help='enables cuda')

parser.add_argument('--save', type=str, default='d:/mcube/python/lstm/model/LSTM_CUBE.pt',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data/train.txt',
                    help='location of the data corpus')
parser.add_argument('--test-data', type=str, default='./data/test.txt',
                    help='location of the data corpus')

parser.add_argument('--dropout', type=float, default=0,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
# parser.add_argument('--embed-dim', type=int, default=64,
#                     help='number of embedding dimension [default: 64]')
# parser.add_argument('--rotate-dim', type=int, default=2,
#                     help='number of rotate data dimension [default:2 ]')
parser.add_argument('--attitude-dim', type=int, default=4,
                    help='number of attitude data dimension [default:4 ]')
parser.add_argument('--dec-hid-dim', type=int, default=512,
                    help='number of decoder hidden dimension [default:512 ]')
parser.add_argument('--enc-hid-dim', type=int, default=512,
                    help='number of encoder hidden dimension [default: 512]')
parser.add_argument('--output-dim', type=int, default=4,
                    help='number of output dimension [default: 4]')
parser.add_argument('--lstm-layers', type=int, default=1,
                    help='biLSTM layer numbers')
parser.add_argument('--bidirectional', action='store_true',
                    help='If True, becomes a bidirectional LSTM [default: False]')
# 2. enc_hid_dim, 3. lstm_layers, 4. dec_hid_dim, 5. dropout**,**6. out_dim**

args = parser.parse_args()
torch.manual_seed(args.seed)

from attention import Attention, Encoder, Decoder, Seq2Seq


use_cuda = torch.cuda.is_available() and args.cuda_able

attn = Attention(args.enc_hid_dim, args.dec_hid_dim)
enc = Encoder(args)
dec = Decoder(args)

model = Seq2Seq(enc, dec)
if use_cuda:
  model = model.cuda()