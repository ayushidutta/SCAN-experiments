import os
import glob

data_dir = 'data'
data_subdirs = ['experiment2CL', 'experiment3CL']

def generate_dataset_cl(file_path):
    src_file = file_path + '.in'
    trg_file = file_path + '.out'
    cl_src_files = [file_path+'_1.in', file_path+'_2.in']
    cl_trg_files = [file_path + '_1.out', file_path + '_2.out']

    with open(src_file, 'r') as fsrc, open(trg_file, 'r') as ftrg:
        src_lines = [x.strip() for x in fsrc.readlines()]
        trg_lines = [x.strip() for x in ftrg.readlines()]

    # Write the cl data to file
    with open(cl_src_files[0], 'w') as c1src, open(cl_trg_files[0], 'w') as c1trg, \
            open(cl_src_files[1], 'w') as c2src, open(cl_trg_files[1], 'w') as c2trg:
        for i, trg_line in enumerate(trg_lines):
            seq_len = len(trg_line.split(' '))
            if seq_len < 10:
                c1src.write(src_lines[i] + "\n")
                c1trg.write(trg_line + "\n")
                c2src.write(src_lines[i] + "\n")
                c2trg.write(trg_line + "\n")
            elif seq_len <=15:
                c2src.write(src_lines[i] + "\n")
                c2trg.write(trg_line + "\n")

# Recurse for all data directories
for i, subdir in enumerate(data_subdirs):
    scan_files = glob.glob(os.path.join(data_dir, subdir, '*train*.in'))
    print(scan_files)
    for scan_file in scan_files:
        file_name = os.path.basename(scan_file).split('.')[0]
        generate_dataset_cl(os.path.join(data_dir, subdir, file_name))
