import argparse
import glob
import os
import json
import subprocess
import sys
import spacy
import random
import argparse
from .dataset_to_json import *

def load_distemist(data_dir):
    # Collect text files
    texts = {}
    for p in glob.glob(os.path.join(data_dir, "text_files/*.txt")):
        fname = os.path.basename(p)
        with open(p, "r", encoding="utf-8") as f:
            texts[fname.split(".")[0]] = f.read()

    # Collect annotations
    ann = {}
    for p in glob.glob(os.path.join(data_dir, "subtrack2_linking/*.tsv")):
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0: #first is the column names
                    continue
                filename, mark, label, off0, off1, span, code, sem_rel  = line.strip().split("\t")
                start = int(off0)
                end = int(off1)
                codes = []
                if "+" in code:
                    codes = code.split("+")
                else:
                    codes = code
                ann.setdefault(filename, []).append({
                    "start": start,
                    "end": end,
                    "text": span,
                    "codes": codes
                })
    return texts, ann

def parse_to_bio(texts, ann, nlp):
    data_out = []
    for fname, text in texts.items():
        doc = nlp(text)
        ents = ann.get(fname, [])
        ents.sort(key=lambda x: x["start"])

        for sent in doc.sents:
            if not sent.text.strip():
                continue
            doc_sent = nlp(sent.text)
            pos_tags = [token.pos_ for token in doc_sent]
            dep_rels = [token.dep_ for token in doc_sent]
            parents = [token.head.i for token in doc_sent] #index of parent token

            tokens = [t.text for t in sent]
            if len(tokens) == 1: # \n novi red "rečenice"
                continue
            tags = [2] * len(tokens)
            codes = []
            entities = []
            
            for e in ents:
                if e["end"] <= sent.start_char or e["start"] >= sent.end_char:
                    continue

                for i, tok in enumerate(sent):
                    tok_start = tok.idx
                    tok_end = tok.idx + len(tok.text)

                    if tok_start == e["start"]:
                        tags[i] = 0 #B-Disease
                    elif tok_start > e["end"]:
                        break

                    if tok_end <= e["start"]:
                        continue
                    if tok_start >= e["end"]:
                        break

                    if tok_start == e["start"]:
                        tags[i] = 0 #B-Disease
                    else:
                        tags[i] = 1 #I-Disease
                
                codes.append(e["codes"])
                entities.append(e["text"])
            
            idx_to_change = []
            idx_to_delete = []
            for i, tag in enumerate(tags): #Ima grešaka u datasetu, krivi span (nekoliko slova promašeno npr. infoma, a ne linfoma je označeno pa dobijemo 2, 1, 1,...)
                if tag == 1:
                    if i == 0 or tags[i-1] == 2:
                        idx_to_change.append(i)
                if tokens[i].strip() == "":
                    idx_to_delete.append(i)

            for i in idx_to_change:
                tags[i] = 0     

            # Izbaciti \n iz tokena
            clean_tags = []
            clean_tokens = []
            for i, (tag, token) in enumerate(zip(tags, tokens)):
                if i not in idx_to_delete:
                    clean_tags.append(tag)
                    clean_tokens.append(token)              

            data_out.append({
                "document_id": fname,
                "sentence": sent.text.strip(),
                "tokens": clean_tokens,
                "tags": clean_tags,
                "codes": codes,
                "entities": entities,
                "pos": pos_tags,
                "dep": dep_rels,
                "parents": parents
            })
    return data_out

def filter_by_text_file(corpus_name, input_file, output_file, sample_ratio, pct_train):
    # Load the full dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Collect unique text files
    document_ids = sorted({item["document_id"] for item in data})
    print(f"Total text files: {len(document_ids)}")

    # Randomly sample args.pct of text files
    sample_size = max(1, int(len(document_ids) * sample_ratio))
    sampled_ids = set(random.sample(document_ids, sample_size))
    print(f"Selected {len(sampled_ids)} text files for the {float(pct_train)*100:.0f}% sample.")

    # Filter all sentences that belong to sampled text files
    filtered_data = [item for item in data if item["document_id"] in sampled_ids]

    # Save to new JSON file
    filtered = output_file + f"{corpus_name}_ner_train_{float(pct_train)*100:.0f}pct.json"
    save_to_json_line(filtered, filtered_data)
    print(f"Saved {len(filtered_data)} sentences to {output_file}")

    return filtered



if __name__=='__main__':
    init_args = argparse_args()
    yaml_args = load_yaml_with_env(init_args.config_file,)
    print(yaml_args)
    args = argparse.Namespace(**yaml_args)

    if not spacy.util.is_package(args.model):
        subprocess.run([sys.executable, "-m", "spacy", "download", args.model])
    nlp = spacy.load(args.model, disable=["ner", "lemmatizer"])
    nlp.add_pipe("sentencizer")

    train_texts, train_ann = load_distemist(args.training)
    train_data = parse_to_bio(train_texts, train_ann, nlp)


    os.makedirs(args.parsed_mesh_folder, exist_ok=True)

    with open(args.parsed_mesh_folder + "distemist_ner.json", "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row) + "\n")

    print("Train JSON saved")

    random.shuffle(train_data)
    split_idx = int(len(train_data) * (1-args.dev_pct))
    train_split = train_data[:split_idx]
    dev_split = train_data[split_idx:]

    with open(args.parsed_mesh_folder + "distemist_ner_train_100pct.json","w",encoding="utf-8") as f:
        for row in train_split:
            f.write(json.dumps(row) + "\n")
    with open(args.parsed_mesh_folder + "distemist_ner_devel.json","w",encoding="utf-8") as f:
        for row in dev_split:
            f.write(json.dumps(row) + "\n")
    print("Train/Dev split saved")

    test_texts, test_ann = load_distemist(args.test)
    test_data = parse_to_bio(test_texts, test_ann, nlp)
    with open(args.parsed_mesh_folder + "distemist_ner_test.json","w",encoding="utf-8") as f:
        for row in test_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Test JSON saved")


    # Filter by text file
    subset_pcts = sorted(args.pcts, reverse=True)
    available_text_files = args.parsed_mesh_folder + "distemist_ner_train_100pct.json"
    previous_pct = 1
    for pct in subset_pcts:
        samples_pct = float(pct) / float(previous_pct) # so that it contains given % from train dataset and not subset it is being extracted from
        filtered_text_files = filter_by_text_file("distemist", available_text_files, args.parsed_mesh_folder, samples_pct, pct)
        available_text_files = filtered_text_files
        previous_pct = pct