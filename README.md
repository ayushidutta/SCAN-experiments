# SCAN-experiments

Final Project for Advanced NLP course, Autumn 2020, University of Copenhagen. 

The code contains 
- Reproduction of the SCAN experiments
- Additional experiments using incorporating linguistic features and Curriculum based learning.

## Requirements
- python3
- pytorch 1.7
- torchtext
- numpy

## Setup and Code details
The 'data' folder contains the SCAN dataset formatted as per the code requirement. 
The 'data_scripts' folder contains the scripts that was used to createt the data in the 'data' folder. 
There are two versions of the code. Updated version are the files suffixed with '_v2'. 
Use '_v2' for all experiments. Use '_v1' only if wanting simpler code for reproducing the SCAN experiments.  

## Run the code

Sample demo bash scripts are available as _demo_task1.sh_,_demo_task2.sh_, _demo_task3.sh_.

To run code from the terminal,execute 
```
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
  --cl=0 \
  --eval=1
```
where,
- _add_pos_: To add POS tags or not, 1 or 0. Default 0.
- _add_dl_: To add Dependency labels or not, 1 or 0. Default 0.
- _cl_: Enable curriculum learning, 1 or 0. Default 0.
- _eval_: Set to eval mode to test model, else it will train, 1 or 0.
