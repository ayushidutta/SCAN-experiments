import os
import glob
import numpy as np

data_dir = 'data/experiment2'
dest_dir = 'data/experiment2CL'
aug_file = os.path.join('data_scripts', 'tasks_test_aug')

scan_file = glob.glob(os.path.join(data_dir, '*train*length.in'))[0]
print(scan_file)
file_name = os.path.basename(scan_file).split('.')[0]
src_file = os.path.join(data_dir, file_name+'.in')
trg_file = os.path.join(data_dir, file_name+'.out')
dest_src_file = os.path.join(dest_dir, file_name+'.in')
dest_trg_file = os.path.join(dest_dir, file_name+'.out')

with open(src_file, 'r') as fsrc, open(trg_file, 'r') as ftrg, open(aug_file + '.in', 'r') as faug_src, \
        open(aug_file + '.out', 'r') as faug_trg, open(dest_src_file, 'w') as wsrc, open(dest_trg_file, 'w') as wtrg:
    src_lines = [x.strip() for x in fsrc.readlines()]
    trg_lines = [x.strip() for x in ftrg.readlines()]
    aug_src_lines = [x.strip() for x in faug_src.readlines()]
    aug_trg_lines = [x.strip() for x in faug_trg.readlines()]
    # Add the data augmentation
    for i, aug_src_line in enumerate(aug_src_lines):
        flag = aug_src_line not in src_lines
        print(f'{aug_src_line} not in src: {aug_src_line not in src_lines}')
        if not flag:
            print(f'index: {src_lines.index(aug_src_line)}')
        if aug_src_line not in src_lines:
            print(f'Added {aug_src_line}!')
            src_lines.append(aug_src_line)
            trg_lines.append(aug_trg_lines[i])
    # Length oversampling
    n_lines = len(trg_lines)
    seq_gt_15 = 0
    max_len_seqs_src = []
    max_len_seqs_trg = []
    for i, trg_line in enumerate(trg_lines):
        seq_len = len(trg_line.split(' '))
        if seq_len > 15:
            seq_gt_15 = seq_gt_15 + 1
            max_len_seqs_src.append(src_lines[i])
            max_len_seqs_trg.append(trg_line)
    seq_lt_15 = n_lines - seq_gt_15
    print(f'No. of <=15 seqs: {seq_lt_15} and >15: {seq_gt_15}')
    s = max(round(seq_lt_15/seq_gt_15)-1, 0)
    print(f'Sampling rate {s}, length of gt_15 lines: {len(max_len_seqs_src)}')
    for i in range(s):
        for j, src_line in enumerate(max_len_seqs_src):
            src_lines.append(src_line)
            trg_lines.append(max_len_seqs_trg[j])
    n_lines = len(src_lines)
    print(f'No of lines final: {n_lines}')
    idxs = np.random.permutation(n_lines)
    for idx in idxs:
        wsrc.write(src_lines[idx] + "\n")
        wtrg.write(trg_lines[idx] + "\n")
