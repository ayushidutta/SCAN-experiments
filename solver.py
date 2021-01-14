import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torchtext.data import BucketIterator
from models import LSTMEncoder, LSTMDecoder,Seq2Seq, GRU_ATTENTIONEncoder, GRU_ATTENTIONDecoder
from tqdm import tqdm
import os
import pdb
from sklearn.metrics import accuracy_score
from data import PAD_TOKEN

CLIP = 5.0

def load_model(SRC, TRG, state):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    # ENC_EMB_DIM = 256
    # DEC_EMB_DIM = 256
    # ENC_HID_DIM = 512
    # DEC_HID_DIM = 512
    # ATTN_DIM = 64
    # ENC_DROPOUT = 0.5
    # DEC_DROPOUT = 0.5

    ENC_HID_DIM = DEC_HID_DIM = state['hidden_dim']
    ENC_EMB_DIM = DEC_EMB_DIM = ENC_HID_DIM
    ENC_DROPOUT = DEC_DROPOUT = state['dropout']

    if state['rnn_type'] == 'lstm':
        enc = LSTMEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        enc = enc.to(device)
        # print(next(enc.parameters()).is_cuda)
        # print('Encoder is on CUDA:{0}'.format(enc.is_cuda))

        dec = LSTMDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, device)
        dec = dec.to(device)
    elif state['rnn_type'] == 'gru':
        enc = GRU_ATTENTIONEncoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        enc = enc.to(device)
        dec = GRU_ATTENTIONDecoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, device)
        dec = dec.to(device)

    model = Seq2Seq(enc, dec, device, state['rnn_type'])
    model = model.to(device)

    # print('Encoder, Decoder, Model is on CUDA: ',enc.device, dec.device, model.device)

    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    PAD_IDX = TRG.vocab.stoi['PAD_TOKEN']

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    return model, optimizer, criterion


def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()

    print('TRAINING !!!')
    # if step_index <= 1000:
    # while step_index <= 100:

    step_index = 0

    for epoch in range(500):

        for batch_index, batch in tqdm(enumerate(iter(iterator))):

            src = batch.src.to(device)
            trg = batch.trg.to(device)

            optimizer.zero_grad()

            output = model(src, trg)

            output = output[1:].view(-1, output.shape[-1])

            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

            optimizer.step()

            step_index += 1

            if step_index % 10000 == 0:
                print('STEP_INDEX:', step_index)

            if step_index >= 100000:
                print('SHOULD BE FINISHED !!!')
                torch.save(model.state_dict(), os.path.join(model_dir, 'best_overall_exp1_p32.pt'))
                return model

        if step_index >= 100000:
            print('SHOULD BE FINISHED !!!')
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_overall_exp1_p32.pt'))
            return model

    torch.save(model.state_dict(), os.path.join(model_dir, 'best_overall_exp1_p32.pt'))
    return model

# TEST MODEL
def test(model, test_iter, eos_index):
    model.eval()
    print(len(test_iter))
    print('EVALUATING !!!')
    correct_count = 0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_iter)):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, train=False, eos_index=eos_index)
            true = list(torch.flatten(trg[1:]))
            if output == true:
                correct_count += 1

    pdb.set_trace()
    print("test done")
    print(correct_count / len(test_iter))
