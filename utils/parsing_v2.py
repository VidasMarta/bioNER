import argparse
import json
import re
import statistics
import string
import numpy as np
import spacy 
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import subprocess
import sys
from pathlib import Path
# from nltk.tokenize.treebank import TreebankWordDetokenizer
from spacy.tokens import Doc
from typing import List, Dict, Tuple

SPACY_NLP = None

def spacy_load_model(model_name: str):
    global SPACY_NLP
    if SPACY_NLP is None:
        SPACY_NLP = get_spacy(model_name)
        return SPACY_NLP
    return SPACY_NLP   

def get_spacy(model_name: str):
    import spacy
    try: return spacy.load(model_name)
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        os.system(f"python3 -m spacy download {model_name}")
        print(f"Downloading {model_name} model")
        import spacy
        return spacy.load(model_name)


def extract_entities(entry):
    tokens = entry["tokens"]
    tags = entry["tags"]

    entities = []
    current_entity = []

    for token, tag in zip(tokens, tags):
        if tag == 0: # B
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
            current_entity.append(token)

        elif tag == 1:  # I
            if current_entity:
                current_entity.append(token)
            else:
                current_entity = [token]

        elif tag == 2:  # O
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []

    if current_entity:
        entities.append(" ".join(current_entity))

    return entities


def load_corpuses(path, nlp, corpus_name):
    corpus_list = []
    id = 0
    if "generated" in corpus_name.lower():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                doc = Doc(nlp.vocab, words=entry["tokens"])
                sentence = doc.text
                entities = extract_entities(entry)
                id += 1
                corpus_list.append((id, sentence, entities, "generated_train", None, entry["term_id"]))
    else:
        with open(path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            for entry in json_data:
                id += 1
                entities = entry["entities"]
                corpus_list.append((id, entry["sentence"], entities, corpus_name, entry["abstract_id"], [None]*len(entities)))       
    
    return corpus_list


def extract_syntax_features(nlp, ids, sentences, entities, corpus_labels, abstract_ids, term_ids,
                            output_path, rewrite):
    features = []
    if os.path.exists(output_path) and not rewrite:
        return 
    for (id, sent, entity, corpus_name, abstract_id, term_id) in tqdm(zip(ids, sentences, entities, corpus_labels, abstract_ids, term_ids)):
        doc = nlp(sent)
        pos_tags = [token.pos_ for token in doc]
        dep_rels = [token.dep_ for token in doc]
        parents = [token.head.i for token in doc] #index of parent token

        features.append({
            "abstract_id": abstract_id,
            "id": id,
            "sentence": doc.text,
            "entities": entity,
            "term_id": term_id,
            "corpus": corpus_name,
            "pos": pos_tags,
            "dep": dep_rels,
            "parents": parents,
            
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for feat in features:
            f.write(json.dumps(feat) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Parsing")
    parser.add_argument('--corpus_name', type=str, required=False, help='Name of the baseline corpus', default="NCBI_train")
    parser.add_argument('--input_file', type=str, required=False, help='Path to train json', default="data/MeSH_NCBI/sm/ncbi_ner_train.json")
    parser.add_argument('--spacy_model', type=str, required=False, help='Name of spacy model', default='en_core_web_sm')  
    parser.add_argument('--output_path_features', type=str, required=False, help='Path where to save output features', default="data/MeSH_NCBI/sm/syntax_features_sent_tree_head.json")
    parser.add_argument('--rewrite', action='store_true', help='Whether to rewrite existing features file')
    parser.add_argument('--gen_pipeline', action='store_true', help='Run as a part of generation pipeline for loading datasets.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for path in [args.input_file, args.output_path_features]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # EXTRACT SYNATX FEATURES FOR BOTH DATASETS
    print("Starting snytax generation:")
    if args.gen_pipeline:
        nlp = spacy_load_model(args.spacy_model)
        
        corpus_list = load_corpuses(args.input_file, nlp, args.corpus_name)
        ids, sentences, entities, corpus_labels, abstract_id, term_id = zip(*corpus_list)

        extract_syntax_features(
            nlp,
            ids, 
            sentences,
            entities,
            corpus_labels,
            abstract_id,
            term_id,
            args.output_path_features,
            args.rewrite
        )


"""    
python3 bioNER/utils/parsing_v2.py \
    --input_file data/ncbi/trf/ncbi_ner_train.json \
    --spacy_model en_core_web_trf \
    --gen_train_path data/ncbi/gen2_json/train.json \
    --output_path_features data/ncbi/trf/syntax_features_sent_tree_head.json

python3 bioNER/utils/parsing_v2.py \
    --input_file /home/mkeber/syn-bioner/bioNER/data/bc5cdr/trf/bc5cdr_ner_train.json \
    --spacy_model en_core_web_trf \
    --output_path_features data/bc5cdr/trf/syntax_features_sent_tree_head.json \
    --gen_pipeline
"""