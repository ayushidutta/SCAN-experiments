#!/bin/bash

DATA_DIR=data
TRAIN_PATH=experiment2CL/tasks_train_length
TEST_PATH=experiment2CL/tasks_test_length
MODEL_DIR=models/exp2cl

python main_v2.py \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --train_path=${TRAIN_PATH} \
  --test_path=${TEST_PATH} \
  --batch_size=1 \
  --hidden_dim=200 \
  --dropout=0.5 \
  --bidirection=0 \
  --num_layers=2 \
  --rnn_type=lstm \
  --add_pos=0 \
  --add_dl=0 \
  --cl=1 \
  --eval=1