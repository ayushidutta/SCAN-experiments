import os
import glob

data_dir = 'data'
data_subdirs = ['experiment2', 'experiment3']
data_aug_file = os.path.join('data_scripts', 'tasks_test_aug')

def generate_dataset_cl(write_file_path, read_file_path):
    read_file_src = read_file_path + '.in'
    read_file_trg = read_file_path + '.out'
    cl_src_files = [write_file_path+'_1.in', write_file_path+'_2.in', write_file_path+'_3.in' ]
    cl_trg_files = [write_file_path + '_1.out', write_file_path + '_2.out', write_file_path + '_3.out']
    seq_len_categ_count = {
        '1': 0, '2': 0, '3': 0
    }
    with open(read_file_src, 'r') as fsrc, open(read_file_trg, 'r') as ftrg, open(data_aug_file+'.in', 'r') as faug_src, \
            open(data_aug_file+'.out', 'r') as faug_trg:
        src_lines = [x.strip() for x in fsrc.readlines()]
        trg_lines = [x.strip() for x in ftrg.readlines()]
        aug_src_lines = [x.strip() for x in faug_src.readlines()]
        aug_trg_lines = [x.strip() for x in faug_trg.readlines()]
    # Add the data augmentation
    for i, aug_src_line in enumerate(aug_src_lines):
        if aug_src_line not in src_lines:
            src_lines.append(aug_src_line)
            trg_lines.append(aug_trg_lines[i])
    del aug_src_lines
    del aug_trg_lines
    # Check seq lengths
    for i, trg_line in enumerate(trg_lines):
        seq_len = len(trg_line.split(' '))
        if seq_len < 10:
            seq_len_categ_count['1'] = seq_len_categ_count['1'] + 1
        elif seq_len <= 16:
            seq_len_categ_count['2'] = seq_len_categ_count['2'] + 1
        else:
            seq_len_categ_count['3'] = seq_len_categ_count['3'] + 1
    sampling_rate = max(round(seq_len_categ_count['2']/seq_len_categ_count['3'])-1, 0)
    # Write the augmented data to file
    with open(cl_src_files[0], 'w') as c1src, open(cl_trg_files[0], 'w') as c1trg, \
            open(cl_src_files[1], 'w') as c2src, open(cl_trg_files[1], 'w') as c2trg, \
            open(cl_src_files[2], 'w') as c3src, open(cl_trg_files[2], 'w') as c3trg :
        for i, trg_line in enumerate(trg_lines):
            c3src.write(src_lines[i]+'\n')
            c3trg.write(trg_line+'\n')
            seq_len = len(trg_line.split(' '))
            if seq_len < 10:
                c1src.write(src_lines[i] + '\n')
                c1trg.write(trg_line + '\n')
                c2src.write(src_lines[i] + '\n')
                c2trg.write(trg_line + '\n')
            elif seq_len <=16:
                c2src.write(src_lines[i] + '\n')
                c2trg.write(trg_line + '\n')
            else:
                for i in range(sampling_rate):
                    c3src.write(src_lines[i] + '\n')
                    c3trg.write(trg_line + '\n')

# Recurse for all data directories
for i, subdir in enumerate(data_subdirs):
    scan_files = glob.glob(os.path.join(data_dir, subdir, '*train*.in'))
    print(scan_files)
    for scan_file in scan_files:
        file_name = os.path.basename(scan_file).split('.')[0]
        generate_dataset_cl(os.path.join(data_dir, subdir+'CL', file_name), os.path.join(data_dir, subdir, file_name))
