#!/bin/bash

DATA_DIR=data
TRAIN_PATH=experiment3/tasks_train_addprim_turn_left
TEST_PATH=experiment3/tasks_test_addprim_turn_left
MODEL_DIR=models/exp3t_gru_att_drop0.1

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
  --rnn_type=gru_attn \
  --add_pos=0 \
  --add_dl=0 \
  --cl=0 \
  --eval=0