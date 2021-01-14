
# -*- coding: utf-8 -*-

import os
import glob

def generate_dataset(path_org,path_trg):
    """
    Function that generates a dataset in the format
    that we will use in the dataloader from torchtext
    input:
        path_org (string): location of the dataset file i
                        in the SCAN format
        path_trg (string): location of where we will save 
                            the new files
    """

    lines = open(path_org,'r').readlines()

    with open(path_trg+".in","w+") as file_in_out, open(path_trg+".out","w+") as file_out_out:
        for line in lines:
            line_list = line.split(':')
            line_in = line_list[1][1:-4]
            line_out = line_list[2][1:]
            file_in_out.write(line_in+"\n")
            file_out_out.write(line_out)

if __name__ == "__main__":

    data_dir = 'data'
    scan_data_files = glob.glob("data/SCAN/*.txt")
    print(scan_data_files)

    for data_file in scan_data_files:
        file_name = os.path.basename(data_file).split('.')[0]
        generate_dataset(data_file, os.path.join(data_dir, file_name))