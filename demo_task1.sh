#!/bin/bash

DATA_DIR=data
TRAIN_PATH=experiment1/tasks_train_simple
TEST_PATH=experiment1/tasks_test_simple
MODEL_DIR=models/exp1_lstm_drop0

python main_v2.py \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --train_path=${TRAIN_PATH} \
  --test_path=${TEST_PATH} \
  --batch_size=1 \
  --hidden_dim=200 \
  --dropout=0 \
  --bidirection=0 \
  --num_layers=2 \
  --rnn_type=lstm \
  --add_pos=1 \
  --add_dl=0 \
  --cl=0 \
  --eval=0