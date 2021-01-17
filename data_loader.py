# -*- coding: utf-8 -*-

# dataloader

import os
from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field, BucketIterator
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
cl_ext = "cl"

def load_data(path_train, path_test, in_ext, out_ext, model_dir, batch_size=1):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	tokenizer = lambda x: x.split()
	lowercase = True
	# We need init_token=BOS_TOKEN, for attn model. For lstm, it doesnt affect much.
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
	print(f"SRC Vocab Freqs: {src.vocab.freqs}")
	print(f"SRC Vocab STOI: {src.vocab.stoi}")
	print(f"SRC Vocab ITOS: {src.vocab.itos}")
	print(f"SRC Vocab Length {len(src.vocab)}")
	print(f"TRG Vocab Freq: {trg.vocab.freqs}")
	print(f"TRG Vocab STOI: {trg.vocab.stoi}")
	print(f"TRG Vocab ITOS: {trg.vocab.itos}")
	print(f"TRG Vocab Length {len(trg.vocab)}")
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
	# We need init_token=BOS_TOKEN, for attn model. For lstm, it doesnt affect much.
	src = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenizer,
					 batch_first=False, lower=lowercase, unk_token=UNK_TOKEN, include_lengths=False)
	trg = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenizer,
					 unk_token=UNK_TOKEN, batch_first=False, lower=lowercase, include_lengths=False)
	exts = ("." + in_ext, "." + out_ext)
	fields = (src, trg)
	data_fields = {
		'src':src,
		'trg':trg
	}
	if add_pos:
		pos = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenizer,
						 batch_first=False, lower=lowercase, unk_token=UNK_TOKEN, include_lengths=False)
		exts = exts + ("."+pos_ext,)
		fields = fields + (pos,)
		data_fields["pos"] = pos
	if add_dl:
		dl = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, tokenize=tokenizer,
						 batch_first=False, lower=lowercase, unk_token=UNK_TOKEN, include_lengths=False)
		exts = exts + ("." + dl_ext,)
		fields = fields + (dl,)
		data_fields["dl"] = dl
	train_data = TranslationDataset(path=path_train, exts=exts, fields=fields)
	test_data = TranslationDataset(path=path_test, exts=exts, fields=fields)
	# build the vocabulary
	src.build_vocab(train_data)
	trg.build_vocab(train_data)
	if add_pos:
		pos.build_vocab(train_data)
	if add_dl:
		dl.build_vocab(train_data)
	print(f"####SRC Vocab Freqs: {src.vocab.freqs}")
	print(f"SRC Vocab STOI: {src.vocab.stoi}")
	print(f"SRC Vocab ITOS: {src.vocab.itos}")
	print(f"SRC Vocab Length {len(src.vocab)}")
	print(f"####TRG Vocab Freq: {trg.vocab.freqs}")
	print(f"TRG Vocab STOI: {trg.vocab.stoi}")
	print(f"TRG Vocab ITOS: {trg.vocab.itos}")
	print(f"TRG Vocab Length {len(trg.vocab)}")
	if add_pos:
		print(f"####POS Vocab Freq: {pos.vocab.freqs}")
		print(f"POS Vocab STOI: {pos.vocab.stoi}")
		print(f"POS Vocab ITOS: {pos.vocab.itos}")
		print(f"POS Vocab Length {len(pos.vocab)}")
	if add_dl:
		print(f"####DL Vocab Freq: {dl.vocab.freqs}")
		print(f"DL Vocab STOI: {dl.vocab.stoi}")
		print(f"DL Vocab ITOS: {dl.vocab.itos}")
		print(f"DL Vocab Length {len(dl.vocab)}")
	return train_data, test_data, data_fields

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

