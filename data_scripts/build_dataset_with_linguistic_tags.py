import os
import glob
import spacy
import io, json

data_dir = 'data_scripts'
data_subdirs = ['']

def generate_dataset(file_path):
    nlp = spacy.load("en_core_web_sm")
    scan_src_file = file_path + ".in"
    scan_trg_file = file_path + ".out"
    scan_json_file = file_path + ".json"
    with open(scan_src_file, 'r') as fsrc, open(scan_trg_file, 'r') as ftrg, open(scan_json_file,"w", encoding='utf-8') as fout:
        src_lines = fsrc.readlines()
        trg_lines = ftrg.readlines()
        for i, src in enumerate(src_lines):
            doc = nlp(src.strip())
            pos_tags = []
            dl_tags = []
            for token in doc:
                pos_tags.append(token.tag_)
                dl_tags.append(token.dep_)
            pos = ' '.join(pos_tags)
            dl = ' '.join(dl_tags)
            data = {"src": src.strip(), "trg": trg_lines[i].strip(), "pos": pos, "dl":dl}
            fout.write(json.dumps(data, ensure_ascii=False)+"\n")

for subdir in data_subdirs:
    scan_files = glob.glob(os.path.join(data_dir, subdir, '*.in'))
    print(scan_files)
    for scan_file in scan_files:
        file_name = os.path.basename(scan_file).split('.')[0]
        generate_dataset(os.path.join(data_dir, subdir, file_name))


