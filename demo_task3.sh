#!/bin/bash

DATA_DIR=data
TRAIN_PATH=experiment3CL/tasks_train_addprim_jump
TEST_PATH=experiment3CL/tasks_test_addprim_jump
MODEL_DIR=models/exp3cl_lstm_att_drop0.1

python main_v2.py \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --train_path=${TRAIN_PATH} \
  --test_path=${TEST_PATH} \
  --batch_size=1 \
  --hidden_dim=100 \
  --dropout=0.1 \
  --bidirection=0 \
  --num_layers=1 \
  --rnn_type=lstm_attn \
  --add_pos=0 \
  --add_dl=0 \
  --cl=1 \
  --eval=1