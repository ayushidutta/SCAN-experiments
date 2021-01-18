import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import pdb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# LSTM Based Models: Encoder -> All sequences, Decoder -> One sequence at a time
class LSTMEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 hid_dim: int,
                 dropout_p: float,
                 bidirection = False,
                 num_layers = 2):
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, bidirectional=bidirection, num_layers=num_layers, dropout=dropout_p)
        self.dropout = nn.Dropout(dropout_p)
        self.bidirectional = bidirection
        self.hidden_dim = hid_dim
        if self.bidirectional:
            self.fc_hidden_bi_to_uni = nn.Linear(hid_dim * 2, hid_dim)
            self.fc_memory_bi_to_uni = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self,
                src: Tensor,
                linguistic_ftrs=None) -> Tuple[Tensor]:
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, memory) = self.rnn(embedded)
        if self.bidirectional: # Project the final bidirectional memory states, as Decoder is unidirectional
            hidden = hidden.view(2, 2, -1, self.hidden_dim)
            memory = memory.view(2, 2, -1, self.hidden_dim)
            hidden = self.fc_hidden_bi_to_uni(torch.cat((hidden[:,0,:,:], hidden[:,1,:,:]), dim=2))
            memory = self.fc_memory_bi_to_uni(torch.cat((memory[:, 0, :, :], memory[:, 1, :, :]), dim=2))
        return outputs, hidden, memory

class LSTMDecoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 hid_dim: int,
                 dropout_p: int,
                 num_layers = 2):
        super(LSTMDecoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=num_layers, dropout=dropout_p)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,
                decoding_input: Tensor,
                decoder_hidden: Tensor,
                decoder_memory: Tensor,
                encoder_output: Tensor) -> Tuple[Tensor]:
        decoding_input = decoding_input.unsqueeze(0)
        embedded = self.dropout(self.embedding(decoding_input))
        output, (decoder_hidden, decoder_memory) = self.rnn(embedded, (decoder_hidden, decoder_memory))
        output = output.squeeze(0)
        output = self.out(output)
        return output, decoder_hidden, decoder_memory

# GRU Based Models
class GRUEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 hid_dim: int,
                 dropout_p: float,
                 bidirection = False,
                 num_layers = 1):
        super(GRUEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_p)
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_dim = hid_dim

    def forward(self,
                src: Tensor,
                linguistic_ftrs = None) -> Tuple[Tensor]:
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

# Code assuming Encoder is unidirectional.
# TODO: Bias within tanh ? Input Concat ? Output concat ? Input state project ?
# output = self.out(torch.cat((output, decoding_input, embedded), dim = 0.0))
# TODO: Code works when batch size=1, have to make it generic
class GRUAttDecoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 hid_dim: int,
                 dropout_p: int,
                 num_layers = 1):
        super(GRUAttDecoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim + hid_dim, hidden_size=hid_dim, num_layers=num_layers, dropout=dropout_p)
        self.w1 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.w2 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,
                decoding_input: Tensor,
                decoder_hidden: Tensor,
                encoder_output: Tensor) -> Tuple[Tensor]:
        decoding_input = decoding_input.unsqueeze(0)
        embedded = self.dropout(self.embedding(decoding_input))
        energy = self.v(torch.tanh(self.w1(decoder_hidden[-1]) + self.w2(encoder_output)))
        attn_weights = F.softmax(energy.view(1, -1), dim=1)
        context = torch.matmul(attn_weights, encoder_output.squeeze(1))
        rnn_input = torch.cat((embedded[0], context), 1).unsqueeze(0)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
        output = output.squeeze(0)
        output = self.out(output)
        return output, decoder_hidden

# SEQ-2-SEQ Model
class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: str,
                 rnn_type: str):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.is_lstm = 'lstm' in rnn_type
        print(f'Is LSTM Model: {self.is_lstm}')

    def forward(self,
                batch: Tensor,
                train: bool = True,
                teacher_forcing_ratio: float = 0.5,
                eos_index: int = 3) -> Tensor:
        src = batch["src"]
        trg = batch["trg"]
        linguistic_ftrs = {}
        if "pos" in batch:
            linguistic_ftrs["pos"] = batch["pos"]
        if "dl" in batch:
            linguistic_ftrs["dl"] = batch["dl"]
        if self.is_lstm:
            encoder_output, hidden, memory = self.encoder(src, linguistic_ftrs=linguistic_ftrs)
        else:
            encoder_output, hidden = self.encoder(src, linguistic_ftrs=linguistic_ftrs)
        input_vec = trg[0, :]
        if train == False:
            outputs = []
            i = 0
            while i < 60:
                if self.is_lstm:
                    output, hidden, memory = self.decoder(input_vec, hidden, memory, encoder_output)
                else:
                    output, hidden = self.decoder(input_vec, hidden, encoder_output)
                top1 = output.argmax(1)
                input_vec = top1
                outputs.append(int(top1))
                i = i+1
                if eos_index == int(top1):
                    return outputs
        else:
            batch_size = trg.shape[-1]
            max_len = trg.shape[0]
            trg_vocab_size = self.decoder.output_dim
            outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
            outputs = outputs.to(self.device)
            for t in range(1, max_len):
                if self.is_lstm:
                    output, hidden, memory = self.decoder(input_vec, hidden, memory, encoder_output)
                else:
                    output, hidden = self.decoder(input_vec, hidden, encoder_output)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.argmax(1)
                input_vec = (trg[t] if teacher_force else top1)
        return outputs