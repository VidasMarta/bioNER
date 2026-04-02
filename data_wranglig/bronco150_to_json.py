import json
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
import spacy 
from collections import defaultdict
import argparse
from .dataset_to_json import argparse_args, save_to_json_line, load_yaml_with_env
# from dataset_to_json import *

def parse_bioc_file(xml_path):
    """
    Parses BioC XML file and returns:
    texts = {doc_id: text}
    annotations = {doc_id: [ {start, end, type} ]}
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    texts = {}
    annotations = {}

    for document in root.iter("document"):
        doc_id = document.find("id").text
        full_text = ""
        ann_list = []

        for passage in document.iter("passage"):
            text_elem = passage.find("text")
            if text_elem is None:
                continue

            passage_text = text_elem.text
            #passage_offset = int(passage.find("offset").text)

            full_text += passage_text

            for ann in passage.iter("annotation"):
                entity_type = None
                for infon in ann.iter("infon"):
                    if infon.attrib.get("key") == "type":
                        entity_type = infon.text

                location = ann.find("location")
                start = int(location.attrib["offset"])
                length = int(location.attrib["length"])
                end = start + length

                ann_list.append({
                    "start": start,
                    "end": end,
                    "type": entity_type
                })

        texts[doc_id] = full_text
        annotations[doc_id] = ann_list

    return texts, annotations


def convert_to_bio(texts, annotations, nlp):
    data = []

    for doc_id, text in texts.items():
        doc = nlp(text)
        ents = annotations.get(doc_id, [])

        for sent in doc.sents:
            if not sent.text.strip():
                continue
            doc_sent = nlp(sent.text)
            pos_tags = [token.pos_ for token in doc_sent]
            dep_rels = [token.dep_ for token in doc_sent]
            parents = [token.head.i for token in doc_sent] #index of parent token

            tokens = [tok.text for tok in sent]
            tags = [2] * len(tokens)
            entities = []

            for ent in ents:
                if ent["type"] != "DIAGNOSIS":
                    continue  # ignore non-DIAGNOSIS entities

                if ent["end"] <= sent.start_char or ent["start"] >= sent.end_char:
                    continue
                
                entity_tokens = []
                for i, tok in enumerate(sent):
                    tok_start = tok.idx
                    tok_end = tok.idx + len(tok.text)

                    if tok_end <= ent["start"]:
                        continue
                    if tok_start >= ent["end"]:
                        break

                    if tok_start == ent["start"]:
                        tags[i] = 0 #B_DIAGNOSIS
                        entity_tokens.append(tok.text)
                    else:
                        tags[i] = 1 # I_DIAGNOSIS
                        entity_tokens.append(tok.text)
                if len(entity_tokens) > 0:
                    entities.append(" ".join(entity_tokens))

            idx_to_delete = []
            for i, tag in enumerate(tags): 
                if tokens[i].strip() == "":
                    idx_to_delete.append(i)

            clean_tags = []
            clean_tokens = []
            for i, (tag, token) in enumerate(zip(tags, tokens)):
                if i not in idx_to_delete:
                    clean_tags.append(tag)
                    clean_tokens.append(token) 
            
            if len(clean_tokens) == 0:
                continue

            data.append({
                "sentence": sent.text.strip(),
                "document_id": doc_id,
                "tokens": clean_tokens,
                "tags": clean_tags, 
                "entities": entities,
                "pos": pos_tags,
                "dep": dep_rels,
                "parents": parents,
            })

    return data

def split_random_sentences(
    input_file,
    output_dir,
    train_ratio=0.7,
    dev_ratio=0.15,
    seed=42,
    prefix="bronco150"
):
    random.seed(seed)

    # Load data
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Split into sentences WITH and WITHOUT entities
    with_ent = []
    without_ent = []

    for item in data:
        if any(tag != 2 for tag in item["tags"]):  # not "O"
            with_ent.append(item)
        else:
            without_ent.append(item)

    print(f"Total sentences: {len(data)}")
    print(f"With entities: {len(with_ent)}")
    print(f"Without entities: {len(without_ent)}")

    def split_list(lst):
        random.shuffle(lst)
        n = len(lst)

        train_size = round(n * train_ratio)
        dev_size = round(n * dev_ratio)

        train_size = max(1, train_size) if n > 0 else 0
        dev_size = max(1, dev_size) if n > 0 else 0

        while train_size + dev_size > n:
            if train_size > dev_size:
                train_size -= 1
            else:
                dev_size -= 1

        test_size = n - train_size - dev_size

        train = lst[:train_size]
        dev = lst[train_size:train_size + dev_size]
        test = lst[train_size + dev_size:]

        return train, dev, test

    # Split both groups
    train_e, dev_e, test_e = split_list(with_ent)
    train_o, dev_o, test_o = split_list(without_ent)

    # Merge back
    train = train_e + train_o
    dev = dev_e + dev_o
    test = test_e + test_o

    # Shuffle final splits
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    print(f"\nFinal splits:")
    print(f"Train: {len(train)}")
    print(f"Dev: {len(dev)}")
    print(f"Test: {len(test)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    for name, split_data in zip(["train", "dev", "test"], [train, dev, test]):
        path = os.path.join(output_dir, f"{prefix}_ner_{name}.json")

        with open(path, "w", encoding="utf-8") as f:
            for row in split_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"{name}: saved to {path}")


def get_train_subset(corpus_name, input_file, output_file, sample_ratio, pct_train, seed = 42):
    random.seed(seed)
    # Load the full dataset
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    data_size = len(data)
    subset_len = int(sample_ratio*data_size)
    train_subset = data[:subset_len]

    # Save to new JSON file
    subset = output_file + f"{corpus_name}_ner_train_{float(pct_train)*100:.0f}pct.json"
    save_to_json_line(subset, train_subset)
    print(f"Saved {len(train_subset)} sentences to {output_file}")

    return subset, train_subset 

def save_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} sentences to {output_path}")


if __name__ == "__main__":
    init_args = argparse_args()
    yaml_args = load_yaml_with_env(init_args.config_file)
    args = argparse.Namespace(**yaml_args)

    if not spacy.util.is_package(args.model):
        subprocess.run([sys.executable, "-m", "spacy", "download", args.model])
    nlp = spacy.load(args.model, disable=["ner", "lemmatizer"])
    nlp.add_pipe("sentencizer")

    os.makedirs(args.parsed_mesh_folder , exist_ok=True)

    #whole dataset
    texts, annotations = parse_bioc_file(args.dataset)
    data  = convert_to_bio(texts, annotations, nlp)
    save_to_json_line(os.path.join(args.parsed_mesh_folder , "bronco150_ner_whole.json"), data)


    split_random_sentences(os.path.join(args.parsed_mesh_folder , "bronco150_ner_whole.json"), args.parsed_mesh_folder)

    # Filter by text file
    filtered  = {}
    subset_pcts = sorted(args.pcts, reverse=True)
    available_text_files = args.parsed_mesh_folder  + "bronco150_ner_train.json"
    previous_pct = 1
    for pct in subset_pcts:
        samples_pct = float(pct) / float(previous_pct) # so that it contains given % from train dataset and not subset it is being extracted from
        filtered_text_files, filtered_data = get_train_subset("bronco150", available_text_files, args.parsed_mesh_folder , samples_pct, pct)
        filtered .setdefault(pct, filtered_data)
        available_text_files = filtered_text_files
        previous_pct = pct


    