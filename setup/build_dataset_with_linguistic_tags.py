import os
import glob
import spacy

data_dir = 'data'
data_subdirs = ['experiment1', 'experiment2', 'experiment3']

def generate_dataset(scan_file_path, out_file_path):
    nlp = spacy.load("en_core_web_sm")
    with open(scan_file_path, 'r') as fin, open(out_file_path+".pos","w+") as fout_pos, open(out_file_path+".dl","w+") as fout_dl:
        lines = fin.readlines()
        for line in lines:
            doc = nlp(line.strip())
            pos_tags = []
            dl_tags = []
            for token in doc:
                pos_tags.append(token.tag_)
                dl_tags.append(token.dep_)
            fout_pos.write(' '.join(pos_tags)+'\n')
            fout_dl.write(' '.join(dl_tags)+'\n')

for subdir in data_subdirs:
    scan_files = glob.glob(os.path.join(data_dir, subdir, '*.in'))
    print(scan_files)
    for scan_file in scan_files:
        file_name = os.path.basename(scan_file).split('.')[0]
        generate_dataset(scan_file, os.path.join(data_dir, subdir, file_name))


