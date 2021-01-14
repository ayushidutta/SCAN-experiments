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
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

def load_data(path_train, path_test, in_ext, out_ext, model_dir, batch_size=1):

	"""
	First attempt at creating a working Dataloader that will
	load the data into objects which can be used by torchtext.
	"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tokenizer = lambda x: x.split()
	lowercase = True

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

	# make iterator for splits
	train_iter = data.BucketIterator(
			repeat=False, sort=False, dataset = train_data,
			batch_size=batch_size, sort_within_batch=True,
			sort_key=lambda x: len(x.src), shuffle=True, device=device)

	# make iterator for splits
	test_iter = data.BucketIterator(
			repeat=False, sort=False, dataset = test_data,
			batch_size=batch_size, sort_within_batch=True,
			sort_key=lambda x: len(x.src), shuffle=True, device=device)

	#pdb.set_trace()


	return train_iter, test_iter, src, trg

