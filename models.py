import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import pdb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super(LSTMEncoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=False, num_layers=2)

        # self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:
        # src_zero = src[0].to(device)
        # embedded = self.dropout(self.embedding(src_zero))
        embedded = self.dropout(self.embedding(src))

        # packed = pack_padded_sequence(embedded, src[1].cpu(),
        #                              batch_first=False)

        outputs, (hidden, memory) = self.rnn(embedded)

        # outputs, _ = pad_packed_sequence(outputs, batch_first=False)

        # hidden = torch.tanh(self.fc(torch.cat((hidden[-1][-2,:,:], hidden[-1][-1,:,:]), dim = 1)))

        # returning the last state of the hidden layer
        return outputs, hidden, memory


class LSTMDecoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 device: str):
        super(LSTMDecoder, self).__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=dec_hid_dim, num_layers=2)

        self.out = nn.Linear(dec_hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,
                decoding_input: Tensor,
                decoder_hidden: Tensor,
                decoder_memory: Tensor) -> Tuple[Tensor]:
        decoding_input = decoding_input.unsqueeze(0)
        decoding_input = decoding_input.to(self.device)

        embedded = self.dropout(self.embedding(decoding_input))

        # output, (decoder_hidden, decoder_memory) = self.rnn(embedded, decoder_hidden.unsqueeze(0))
        # pdb.set_trace()
        output, (decoder_hidden, decoder_memory) = self.rnn(embedded, (decoder_hidden, decoder_memory))

        # print('\n\n\n\nhehelolz\n\n\n\n')

        # embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        # decoding_input = decoding_input.squeeze(0)

        # output = self.out(torch.cat((output, decoding_input, embedded), dim = 1.0))
        output = self.out(output)

        return output, decoder_hidden, decoder_memory


class GRU_ATTENTIONEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super(GRU_ATTENTIONEncoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=False, num_layers=1)

        # self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:
        # src_zero = src[0].to(device)
        # embedded = self.dropout(self.embedding(src_zero))
        embedded = self.dropout(self.embedding(src))

        # packed = pack_padded_sequence(embedded, src[1].cpu(),
        #                              batch_first=False)

        outputs, hidden = self.rnn(embedded)

        # outputs, _ = pad_packed_sequence(outputs, batch_first=False)

        # hidden = torch.tanh(self.fc(torch.cat((hidden[-1][-2,:,:], hidden[-1][-1,:,:]), dim = 1)))

        # returning the last state of the hidden layer
        return outputs, hidden


class GRU_ATTENTIONDecoder(nn.Module):

    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 device: str):
        super(GRU_ATTENTIONDecoder, self).__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(input_size=emb_dim * 2, hidden_size=dec_hid_dim, num_layers=1)

        self.energy = nn.Linear(dec_hid_dim * 2, 1)

        self.out = nn.Linear(dec_hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self,
                decoding_input: Tensor,
                decoder_hidden: Tensor,
                encoder_output: Tensor) -> Tuple[Tensor]:
        decoding_input = decoding_input.unsqueeze(0)
        decoding_input = decoding_input.to(self.device)

        sequence_length = encoder_output.shape[0]

        embedded = self.dropout(self.embedding(decoding_input))

        hidden_reshaped = decoder_hidden.repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(F.tanh(torch.cat((hidden_reshaped, encoder_output), dim=2))))

        attention = self.softmax(energy)

        attention = attention.permute(1, 2, 0)
        encoder_output = encoder_output.permute(1, 0, 2)

        context_vector = torch.bmm(attention, encoder_output).permute(1, 0, 2)

        rnn_input = torch.cat((context_vector, embedded), dim=2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)

        output = output.squeeze(0)

        output = self.out(output)

        return output, decoder_hidden


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
        self.rnn_type = rnn_type

    def forward(self,
                src: Tensor,
                trg: Tensor,
                train: bool = True,
                teacher_forcing_ratio: float = 0.5,
                eos_index: int = 3) -> Tensor:

        # batch_size = src[0].shape[1]
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
        outputs = outputs.to(self.device)

        if self.rnn_type == 'lstm':
            _, hidden, memory = self.encoder(src)
        elif self.rnn_type == 'gru':
            encoder_output, hidden = self.encoder(src)

        # first input_vec to the decoder is the <sos> token
        # output = trg[:,0]
        # input_vec = trg[:,0]
        input_vec = trg[0, :]

        if train == False:

            outputs = []
            i = 0

            nonstop = True
            while nonstop:
                if self.rnn_type == 'lstm':
                    output, hidden, memory = self.decoder(input_vec, hidden, memory)
                elif self.rnn_type == 'gru':

                    output, hidden = self.decoder(input_vec, hidden, encoder_output)
                # outputs[t] = output
                # pdb.set_trace()
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.argmax(1)
                input_vec = top1
                outputs.append(int(top1))
                if eos_index == int(top1):
                    return outputs
                if i > 60:
                    # stop when model predict very long sequences
                    return outputs

        for t in range(1, max_len):
            if self.rnn_type == 'lstm':
                output, hidden, memory = self.decoder(input_vec, hidden, memory)
            elif self.rnn_type == 'gru':
                output, hidden = self.decoder(input_vec, hidden, encoder_output)
            # outputs[:,t] = output
            # pdb.set_trace()
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            if train == True:
                # input_vec = (trg[:,t] if teacher_force else top1)
                input_vec = (trg[t] if teacher_force else top1)
            else:
                input_vec = top1

        return outputs