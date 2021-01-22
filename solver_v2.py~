import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torchtext.data import BucketIterator
from models_v2 import LSTMEncoder, LSTMDecoder, Seq2Seq, GRUEncoder, GRUAttDecoder, LSTMAttDecoder
from tqdm import tqdm
import os
import pdb
from sklearn.metrics import accuracy_score
from data_loader import PAD_TOKEN
import matplotlib.pyplot as plt

CLIP = 5.0

def load_model(data_fields, state):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRC = data_fields["src"][1]
    TRG = data_fields["trg"][1]
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    EMB_DIM = HID_DIM = state['hidden_dim']
    DROPOUT = state['dropout']
    add_linguistic_ftrs = state["add_linguistic_ftrs"]
    if add_linguistic_ftrs['add_pos']:
        add_linguistic_ftrs['pos_dim'] = len(data_fields["pos"][1].vocab)
    if add_linguistic_ftrs['add_dl']:
        add_linguistic_ftrs['dl_dim'] = len(data_fields["dl"][1].vocab)
    enc_models = {
        'lstm': LSTMEncoder,
        'gru': GRUEncoder,
        'lstm_attn': LSTMEncoder,
        'gru_attn': GRUEncoder
    }
    dec_models = {
        'lstm': LSTMDecoder,
        'gru': GRUAttDecoder,
        'lstm_attn': LSTMAttDecoder,
        'gru_attn': GRUAttDecoder
    }
    enc_model = enc_models[state['rnn_type']]
    enc = enc_model(INPUT_DIM, EMB_DIM, HID_DIM, DROPOUT, bidirection=state['bidirection'], num_layers=state['num_layers'], add_linguistic_ftrs=add_linguistic_ftrs)
    enc = enc.to(device)
    dec_model = dec_models[state['rnn_type']]
    dec = dec_model(OUTPUT_DIM, EMB_DIM, HID_DIM, DROPOUT, num_layers=state['num_layers'])
    dec = dec.to(device)
    model = Seq2Seq(enc, dec, device, state['rnn_type'])
    model = model.to(device)
    #print('Encoder, Decoder, Model is on CUDA: ',enc.device, dec.device, model.device)

    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    # Use default weight initialisation by pytorch
    # model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def count_parameters(model: nn.Module):
        for name, p in model.named_parameters():
            print(f'Param: {name}, grad {p.requires_grad}, size {p.size()}, {p.numel()}')
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    PAD_IDX = TRG.vocab.stoi['PAD_TOKEN']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    return model, optimizer, criterion

# TRAIN MODEL
def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          model_dir: str,
          add_linguistic_ftrs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    step_index = 0
    step_loss = 0
    step_loss_values = []
    while step_index < 100000:
        for batch_index, batch in tqdm(enumerate(iter(iterator))):
            src = batch.src.to(device)
            trg = batch.trg.to(device)
            linguistic_ftrs = {}
            if add_linguistic_ftrs is not None:
                if add_linguistic_ftrs["add_pos"]:
                    linguistic_ftrs["pos"] = batch.pos.to(device)
                if add_linguistic_ftrs["add_dl"]:
                    linguistic_ftrs["dl"] = batch.dl.to(device)
            optimizer.zero_grad()
            output = model(src, trg, linguistic_ftrs)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            step_index += 1
            step_loss += loss.item()
            if step_index % 1000 == 0:
                print('Loss/1K steps:', step_loss / 1000)
                step_loss_values.append(step_loss / 1000)
                step_loss = 0
            if step_index % 100000 == 0:
                torch.save(model.state_dict(), os.path.join(model_dir, 'model_' + str(step_index) + '.pt'))
    # Save training loss plot
    plot_loss(step_loss_values, save_dir=model_dir)
    return model

# TRAIN BY Curriculum Learning
def train_by_cl(model: nn.Module,
          iterators,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          model_dir: str,
          add_linguistic_ftrs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    step_index = 0
    step_loss = 0
    step_loss_values = []
    iter_epochs = [30, 30, 1, 4]
    epoch = 0
    for iter_idx in range(1):
        for _ in range(iter_epochs[iter_idx]):
            for batch_index, batch in tqdm(enumerate(iter(iterators[iter_idx]))):
                src = batch.src.to(device)
                trg = batch.trg.to(device)
                linguistic_ftrs = {}
                if add_linguistic_ftrs is not None:
                    if add_linguistic_ftrs["add_pos"]:
                        linguistic_ftrs["pos"] = batch.pos.to(device)
                    if add_linguistic_ftrs["add_dl"]:
                        linguistic_ftrs["dl"] = batch.dl.to(device)
                optimizer.zero_grad()
                output = model(src, trg, linguistic_ftrs)
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
                step_index += 1
                step_loss += loss.item()
                if step_index % 1000 == 0:
                    print('Loss/1K steps:', step_loss / 1000)
                    step_loss_values.append(step_loss / 1000)
                    step_loss = 0
                if step_index % 100000 == 0:
                    torch.save(model.state_dict(), os.path.join(model_dir, 'model_' + str(step_index) + '.pt'))
                    print('Reached Max steps! phew!')
                    plot_loss(step_loss_values, save_dir=model_dir)
                    return model
            epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(model_dir, 'model_e' + str(epoch) + '.pt'))
    # Save training loss plot
    print(f'Training done, epochs: {epoch}')
    plot_loss(step_loss_values, save_dir=model_dir)
    return model

# TEST MODEL
def test(model, test_iter, eos_index, add_linguistic_ftrs=None, data_fields=None):
    model.eval()
    print(f'Evaluating Test data of size: {len(test_iter)}')
    correct_count = 0
    qual_res = {}
    save = False
    if data_fields is not None:
        src_vocab = data_fields["src"][1].vocab
        trg_vocab = data_fields["trg"][1].vocab
        save = True
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_iter)):
            src = batch.src
            trg = batch.trg
            linguistic_ftrs = {}
            if add_linguistic_ftrs is not None:
                if add_linguistic_ftrs["add_pos"]:
                    linguistic_ftrs["pos"] = batch.pos
                if add_linguistic_ftrs["add_dl"]:
                    linguistic_ftrs["dl"] = batch.dl
            output = model(src, trg, linguistic_ftrs, train=False, eos_index=eos_index)
            true = list(torch.flatten(trg[1:]))
            if output == true:
                correct_count += 1
            elif save:
                cmd = ''
                for i in list(torch.flatten(src)):
                    cmd = cmd + ' ' + src_vocab.itos[i]
                gt = ''
                for i in list(torch.flatten(trg)):
                    gt = gt + ' ' + trg_vocab.itos[i]
                pred = ''
                for i in output:
                    pred = pred + ' ' + trg_vocab.itos[i]
                qual_res[batch_idx] = {'cmd': cmd.strip(), 'pred': pred.strip(), 'gt': gt.strip()}

    print("Accuracy: ")
    print((correct_count / len(test_iter))*100.0)
    return qual_res

# PLOTS
def plot_loss(loss_step_values, save_dir=None):
    steps =  [(x+1)*1000 for x in range(len(loss_step_values))]
    plt.plot(steps, loss_step_values)
    if save_dir is not None:
        print(f"Loss values:{loss_step_values}")
        plt.savefig(os.path.join(save_dir, 'Training_loss.png'))