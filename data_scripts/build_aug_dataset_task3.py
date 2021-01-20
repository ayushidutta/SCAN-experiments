import os
import glob
import numpy as np

data_dir = 'data/experiment3'
dest_dir = 'data/experiment3CL'
aug_file = os.path.join('data_scripts', 'tasks_test_aug')

scan_file = glob.glob(os.path.join(data_dir, '*train*jump.in'))[0]
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
    jump_src_lines = []
    jump_trg_lines = []
    print(len(src_lines))
    for i, aug_src_line in enumerate(aug_src_lines):
        flag = aug_src_line not in src_lines
        print(f'{aug_src_line} in src: {aug_src_line not in src_lines}')
        if not flag:
            print(f'index: {src_lines.index(aug_src_line)}')
        if aug_src_line not in src_lines:
            if "jump" in aug_src_line:
                jump_src_lines.append(aug_src_line)
                jump_trg_lines.append(aug_trg_lines[i])
            else:
                print(f'Added {aug_src_line}!')
                src_lines.append(aug_src_line)
                trg_lines.append(aug_trg_lines[i])
    print(len(src_lines))
    # Oversample "Jump" keyword
    n_lines = len(src_lines)
    n_jump_lines = len(jump_src_lines)
    s = int(n_lines / (9*n_jump_lines))
    for i in range(s):
        for j, src_line in enumerate(jump_src_lines):
            src_lines.append(src_line)
            trg_lines.append(jump_trg_lines[j])
    n_lines = len(src_lines)
    for i, src_line in enumerate(src_lines):
        wsrc.write(src_line+"\n")
        wtrg.write(trg_lines[i]+"\n")







