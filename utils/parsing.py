# TODO: delete script
import argparse
import spacy
import json
import numpy as np
from pathlib import Path
from nltk.tokenize.treebank import TreebankWordDetokenizer # TODO: 
from typing import List, Dict, Tuple
import subprocess
import sys
import os

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


def load_json_corpus(path, corpus_name):
    triple = []
    detok = TreebankWordDetokenizer()
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        for entry in json_data:
            text = detok.detokenize(entry["tokens"])
            entity = extract_entities(entry)
            triple.append((text, entity, corpus_name))
    return triple

def load_json_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def extract_syntax_features(model, sentences, entities, corpus_labels, output_path):
    features = []
    nlp = spacy.load(model)
    for i, (sent, entity, corpus_name) in enumerate(zip(sentences, entities, corpus_labels), start=1):
        doc = nlp(sent)
        pos_tags = [token.pos_ for token in doc]
        dep_rels = [token.dep_ for token in doc]
        parents = [token.head.i for token in doc] #index of parent token

        features.append({
            "id": i,
            "sentence": doc.text,
            "entities": entity,
            "corpus": corpus_name,
            "pos": pos_tags,
            "dep": dep_rels,
            "parents": parents
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)



def build_global_mappings_json(dataset: List[Dict]) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    """Compute max_len, relation and POS tag vocabularies across the corpus."""
    max_len = 0
    rel_set, pos_set = set(), set()
    for ex in dataset:
        deps = ex.get("dep", [])
        poss = ex.get("pos", [])
        max_len = max(max_len, len(deps))
        rel_set.update(deps)
        pos_set.update(poss)
    rel_to_id = {r: i for i, r in enumerate(sorted(rel_set))}
    pos_to_id = {p: i for i, p in enumerate(sorted(pos_set))}
    return max_len, rel_to_id, pos_to_id

def encode_json_dependency_data(
    dataset: List[Dict],
    pad_value: int = -1
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int], int]:
    """
    Encode dataset where each token is represented by:
      (parent_index, relation_id, node_type_id)
    Uses global vocabularies and padding.
    """
    max_len, rel_to_id, pos_to_id = build_global_mappings_json(dataset)

    encoded_data = []
    for ex in dataset:
        heads = ex.get("parents", [])  # now actually head indices
        deps = ex.get("dep", [])
        pos_tags = ex.get("pos", [])
        sent_id = ex.get("id", None)
        corpus = ex.get("corpus", None)
        n = len(heads)

        parent_vec = np.full(max_len, pad_value, dtype=int)
        rel_vec = np.full(max_len, pad_value, dtype=int)
        pos_vec = np.full(max_len, pad_value, dtype=int)

        for i in range(n):
            head = heads[i]
            # Convert head to -8 for roots (self or 0)
            parent_vec[i] = -8 if head == 0 or head == i else head
            rel_vec[i] = rel_to_id[deps[i]]
            pos_vec[i] = pos_to_id[pos_tags[i]]

        encoding = np.stack([parent_vec, rel_vec, pos_vec], axis=1).flatten().tolist()

        encoded_data.append({
            "id": sent_id,
            "corpus": corpus,
            "encoding": encoding
        })

    return encoded_data, rel_to_id, pos_to_id, max_len


def pretty_print_encoding(enc_vector: np.ndarray, max_len: int):
    """Pretty-print per-token triplets."""
    triplets = enc_vector.reshape(max_len, 3)
    for i, (p, r, t) in enumerate(triplets):
        print(f"  token {i+1:>2}: parent={p:>3}, rel_id={r:>3}, pos_id={t:>3}")

def embedd_morpho_syntax(fname, output_path_emb, output_path_dict):
    dataset = load_json_data(fname)
    encodings, rel_map, pos_map, max_len = encode_json_dependency_data(dataset)


    print("=== Global statistics ===")
    print(f"Max sentence length: {max_len}")
    print(f"Unique dependency relations ({len(rel_map)}): {sorted(rel_map.keys())}")
    print(f"Unique POS tags ({len(pos_map)}): {sorted(pos_map.keys())}")

    with open(output_path_emb, "w", encoding="utf-8") as f:
        json.dump(encodings, f, indent=2, ensure_ascii=False)

    print(f"Saved encoded dataset to {output_path_emb}")

    save_dict = {
        "rel_to_id": rel_map,
        "pos_to_id": pos_map,
        "max_len": max_len
    }

    with open(output_path_dict, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved encoded dataset to {output_path_dict}")


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--ncbi_train_path', type=str, required=False, help='Path to where NCBI train json is saved', default="bioNER/data/ncbi/ncbi/train.json")
    parser.add_argument('--gen_train_path', type=str, required=False, help='Path to where generated train json is saved', default="bioNER/data/ncbi/gen2_json/train.json")
    parser.add_argument('--output_path_features', type=str, required=False, help='Path where to save output features', default="bioNER/data/ncbi/syntax_features/syntax_features_sent_tree_head.json")
    parser.add_argument('--output_path_embeddings', type=str, required=False, help='Path where to save output embeddings', default="bioNER/data/ncbi/syntax_features/morpho_syntax_features.json")
    parser.add_argument('--output_path_dict', type=str, required=False, help='Path where to save output dict', default="bioNER/data/ncbi/syntax_features/dict.json")
    parser.add_argument('--model', type=str, required=False, help='Name of spacy model', default='en_core_web_trf') #trf    
    return parser.parse_args()
"""
python3 bioNER/utils/parsing.py --ncbi_train_path bioNER/data/ncbi/trf/ncbi_ner_train.json \
    --gen_train_path bioNER/data/ncbi/gen2_json/train.json \
    --output_path_features bioNER/data/ncbi/trf/syntax_features/syntax_features_sent_tree_head.json \
    --output_path_embeddings bioNER/data/ncbi/trf/syntax_features/morpho_syntax_features.json \
    --output_path_dict bioNER/data/ncbi/trf/syntax_features/dict.json \
    --model en_core_web_trf

python3 bioNER/utils/parsing.py --ncbi_train_path bioNER/data/ncbi/lg/ncbi_ner_train.json \
    --gen_train_path bioNER/data/ncbi/gen2_json/train.json \
    --output_path_features bioNER/data/ncbi/lg/syntax_features/syntax_features_sent_tree_head.json \
    --output_path_embeddings bioNER/data/ncbi/lg/syntax_features/morpho_syntax_features.json \
    --output_path_dict bioNER/data/ncbi/lg/syntax_features/dict.json \
    --model en_core_web_lg
"""
    
    
def extract_args():
    args = parse_args()
    model = args.model
    ncbi_path = args.ncbi_train_path
    gen_path = args.gen_train_path
    output_path_feat = args.output_path_features
    output_path_emb = args.output_path_embeddings
    output_path_dict = args.output_path_dict
    return model, ncbi_path, gen_path, output_path_feat, output_path_emb, output_path_dict

if __name__ == "__main__":
    model, ncbi_path, gen_path, output_path_feat, output_path_emb, output_path_dict = extract_args()
    try:
        spacy.load(model)
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", model])
        import spacy
        
    for path in [output_path_feat, output_path_emb, output_path_dict]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    NCBI_train = Path(ncbi_path)
    generated_train = Path(gen_path)

    triple = load_json_corpus(NCBI_train, "NCBI_train") + load_json_corpus(generated_train, "generated_train")
    sentences, entities, corpus_labels = zip(*triple)
    print(f"Loaded {len(sentences)} sentences: "
        f"{corpus_labels.count('NCBI_train')} from NCBI_train and "
        f"{corpus_labels.count('generated_train')} from Generated.")
    
    extract_syntax_features(
        model,
        sentences,
        entities,
        corpus_labels,
        output_path_feat
    )

    embedd_morpho_syntax(
        output_path_feat, 
        output_path_emb, output_path_dict
    )
