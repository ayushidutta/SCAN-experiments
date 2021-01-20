# -*- coding: utf-8 -*-

# dataloader

import os
from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field, TabularDataset, BucketIterator
import dill
import pdb
import torch

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
in_ext = "in"
out_ext = "out"
pos_ext = "pos"
dl_ext = "dl"

def load_data(path_train, path_test, in_ext, out_ext, model_dir, batch_size=1):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer = lambda x: x.split()
	lowercase = True
	# TODO: Whether we need init_token=BOS_TOKEN, is still questionable. With POS tags,
	#  it was needed, for lstm didnt matter, for gru attn model, absolutely not.
	src = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
						   pad_token=PAD_TOKEN, tokenize=tokenizer,
						   batch_first=False, lower=lowercase,
						   unk_token=UNK_TOKEN,
						   include_lengths=False)
	trg = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
						   pad_token=PAD_TOKEN, tokenize=tokenizer,
						   unk_token=UNK_TOKEN,
						   batch_first=False, lower=lowercase,
						   include_lengths=False)
	train_data = TranslationDataset(path=path_train,
										exts=("." + in_ext, "." + out_ext),
										fields=(src, trg))
	test_data = TranslationDataset(path=path_test,
										exts=("." + in_ext, "." + out_ext),
										fields=(src, trg))
	# build the vocabulary
	src.build_vocab(train_data)
	trg.build_vocab(train_data)
	print_vocab(src, trg)
	with open(os.path.join(model_dir, "src.Field"), "wb") as f:
		dill.dump(src, f)
	with open(os.path.join(model_dir, "trg.Field"), "wb") as f:
		dill.dump(trg, f)
	# Make sure we use, sort=False else accuracy drops
	train_iter = data.BucketIterator(
			repeat=False, sort=False, dataset = train_data,
			batch_size=batch_size, sort_within_batch=True,
			sort_key=lambda x: len(x.src), shuffle=True, train=True, device=device)
	test_iter = data.BucketIterator(
			repeat=False, sort=False, dataset = test_data,
			batch_size=1, sort_within_batch=False,
			sort_key=lambda x: len(x.src), shuffle=False, train=False, device=device)
	#pdb.set_trace()
	return train_iter, test_iter, src, trg

def load_data_v2(path_train, path_test, model_dir, add_pos=False, add_dl=False):
	tokenizer = lambda x: x.split()
	lowercase = True
	# TODO: Whether we need init_token=BOS_TOKEN, is still questionable. With POS tags,
	#  it was needed, for lstm , didn't matter, for gru attn model, absolutely not.
	src_init_token=BOS_TOKEN
	src = data.Field(init_token=src_init_token, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenizer,
					 batch_first=False, lower=lowercase, unk_token=UNK_TOKEN, include_lengths=False)
	trg = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenizer,
					 unk_token=UNK_TOKEN, batch_first=False, lower=lowercase, include_lengths=False)
	pos = dl= None
	fields = {
		"src":("src", src),
		"trg": ("trg", trg),
	}
	if add_pos:
		pos = data.Field(init_token=src_init_token, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenizer,
						 batch_first=False, lower=lowercase, unk_token=UNK_TOKEN, include_lengths=False)
		fields["pos"] = ("pos", pos)
	if add_dl:
		dl = data.Field(init_token=src_init_token, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenizer,
						 batch_first=False, lower=lowercase, unk_token=UNK_TOKEN, include_lengths=False)
		fields["dl"] = ("dl", dl)
	train_data, test_data = TabularDataset.splits( path='', train= path_train + '.json', test=path_test+'.json',
								format='json', fields=fields)
	# build the vocabulary
	src.build_vocab(train_data)
	trg.build_vocab(train_data)
	if add_pos:
		pos.build_vocab(train_data)
	if add_dl:
		dl.build_vocab(train_data)
	print_vocab(src, trg, pos, dl)
	return train_data, test_data, fields

def get_data_iters(train_data, test_data, batch_size=1, cl=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	data_iters = []
	if cl:
		pass
	else:
		# Make sure we use, sort=False else accuracy drops
		train_iter = data.BucketIterator(repeat=False, sort=False, dataset=train_data, batch_size=batch_size, sort_within_batch=True,
			sort_key=lambda x: len(x.src), shuffle=True, train=True, device=device)
		data_iters.append(train_iter)
	test_iter = data.BucketIterator(repeat=False, sort=False, dataset=test_data, batch_size=1, sort_within_batch=False,
		sort_key=lambda x: len(x.src), shuffle=False, train=False, device=device)
	data_iters.append(test_iter)
	return data_iters

def load_data_and_iter_cl(path_train, path_test, model_dir):
	tokenizer = lambda x: x.split()
	lowercase = True
	# TODO: Whether we need init_token=BOS_TOKEN, is still questionable. With POS tags,
	#  it was needed, for lstm , didn't matter, for gru attn model, absolutely not.
	src_init_token = BOS_TOKEN
	src = data.Field(init_token=src_init_token, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenizer,
					 batch_first=False, lower=lowercase, unk_token=UNK_TOKEN, include_lengths=False)
	trg = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenizer,
					 unk_token=UNK_TOKEN, batch_first=False, lower=lowercase, include_lengths=False)
	fields = {
		"src": ("src", src),
		"trg": ("trg", trg),
	}
	train_data, test_data = TabularDataset.splits(path='', train=path_train + '.json', test=path_test + '.json',
												  format='json', fields=fields)
	# build the vocabulary
	src.build_vocab(train_data)
	trg.build_vocab(train_data)
	print_vocab(src, trg)

	train_data_list = []
	for i in range(3):
		x_train = TabularDataset(path=path_train+'_'+str(i)+'.json', format='json', fields=fields)
		train_data_list.append(x_train)
	train_data_list.append(train_data)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Make sure we use, sort=False else accuracy drops
	data_iters = []
	for i in range(4):
		train_iter = data.BucketIterator(repeat=False, sort=False, dataset=train_data_list[i], batch_size=1,
									 sort_within_batch=True, sort_key=lambda x: len(x.src), shuffle=True, train=True, device=device)
		data_iters.append(train_iter)

	test_iter = data.BucketIterator(repeat=False, sort=False, dataset=test_data, batch_size=1, sort_within_batch=False,
								sort_key=lambda x: len(x.src), shuffle=False, train=False, device=device)
	data_iters.append(test_iter)
	return data_iters, fields

def print_vocab(src, trg, pos=None, dl=None):
	print(f"####SRC Vocab Freqs: {src.vocab.freqs}")
	print(f"SRC Vocab STOI: {src.vocab.stoi}")
	print(f"SRC Vocab ITOS: {src.vocab.itos}")
	print(f"SRC Vocab Length {len(src.vocab)}")
	print(f"####TRG Vocab Freq: {trg.vocab.freqs}")
	print(f"TRG Vocab STOI: {trg.vocab.stoi}")
	print(f"TRG Vocab ITOS: {trg.vocab.itos}")
	print(f"TRG Vocab Length {len(trg.vocab)}")
	if pos is not None:
		print(f"####POS Vocab Freq: {pos.vocab.freqs}")
		print(f"POS Vocab STOI: {pos.vocab.stoi}")
		print(f"POS Vocab ITOS: {pos.vocab.itos}")
		print(f"POS Vocab Length {len(pos.vocab)}")
	if dl is not None:
		print(f"####DL Vocab Freq: {dl.vocab.freqs}")
		print(f"DL Vocab STOI: {dl.vocab.stoi}")
		print(f"DL Vocab ITOS: {dl.vocab.itos}")
		print(f"DL Vocab Length {len(dl.vocab)}")


