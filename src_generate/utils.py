import re
import string
import json
import os
import numpy as np
import obonet
from typing import List, Dict, Any, Optional, Tuple
import argparse
import pandas as pd
import subprocess
import sys


SPACY_NLP = None


def remove_code_fences(text: str) -> str:
    return re.sub(r'```[\w]*\n?', '', text).strip()

def clean_text(text: str) -> str:
    text = re.sub(r'```+', '', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?,])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<\|.*?\|>', '', text)
    return text.strip()

def check_last_token(tokens_lower: List[str]) -> List[str]:
    tokens_lower[-1] = tokens_lower[-1].rstrip(string.punctuation)
    return tokens_lower

def get_diseases(args: argparse.Namespace):
    graph = obonet.read_obo(args.obo_file_path)
    do_terms = [(data["name"], node) for node, data in graph.nodes(data=True) if "name" in data]
    if args.verbose:
        print(f"Number of nodes (terms): {graph.number_of_nodes()}")
        print(f"Number of edges (relations): {graph.number_of_edges()}")
        print('First 20 terms: ', do_terms[:20])  # Show first 20 terms
    return do_terms

def create_concept_txt_file(file_path = '/home/mkeber/syn-bioner/data/SNOMEDCT/CONCEPT.csv',
                            output_path = '/home/mkeber/syn-bioner/data/SNOMEDCT/concepts.txt'):
    concept_df = pd.read_csv(file_path, 
                         delimiter='\t', on_bad_lines='skip', dtype=str)
    filtered_df = concept_df[concept_df.domain_id == 'Condition'].copy()
    concepts = filtered_df[filtered_df.concept_class_id.isin(['Disorder', 'Clinical Finding',
                'ICD10 code', 'Context-dependent', 'Event', 'Morph Abnormality'])
                ].concept_name.unique()
    with open(output_path, 'w') as f:
        for concept in concepts:
            f.write(f"{concept}\n")

def load_training_samples(
    file_path: str
    ) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_kshot_block(examples: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    formatted = []
    for ex in examples:
        ex_text = ex.get("sentence", "").strip()
        entities = ", ".join(ex.get("entities", []))
        block = f"Sentence: {ex_text}\nEntities: [{entities}]"
        if args.include_pos:
            pos_tags = " ".join(ex.get("pos", []))
            block += f"\nPOS: {pos_tags}"
        if args.include_dep:
            deps = " ".join(ex.get("dep", []))
            block += f"\nDEP: {deps}"

        formatted.append(block + "\n")
    return "\n".join(formatted)

def sample_k_examples(args: argparse.Namespace, kshot_pool):
    k = min(args.kshot_size, len(kshot_pool))
    sampled = np.random.choice(kshot_pool, size=k, replace=False)
    kshot_text_block = format_kshot_block(sampled, args)
    used_ids = [ex.get("id", f"ex_{idx}") for idx, ex in enumerate(sampled)]
    print(f"Sampled k-shot example IDs: {used_ids}")
    return kshot_text_block, used_ids

# TODO: parsing Sentence: ... Entities: [...] format
def parse_text_entities_format(args: argparse.Namespace, text: str) -> Tuple[str, List[str]]:
    pattern = r'Sentence:\s*(.*?)\s*Entities:\s*\[(.*?)\]'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        sentence = match.group(1).strip()
        entities_str = match.group(2).strip()
        entities = [ent.strip() for ent in entities_str.split(',')] if entities_str else []
        return sentence, entities
    else:
        print("[WARNING] Could not parse the generated text for sentence and entities.")
        return text, []
    

def get_spacy_model(model: str):
    global SPACY_NLP
    if SPACY_NLP is None:
        try:
            import spacy
            SPACY_NLP = spacy.load(model)
        except OSError:
            subprocess.run(["python3", "-m", "spacy", "download", model])
            import spacy
            SPACY_NLP = spacy.load(model)
    else:
        return SPACY_NLP
        
"""python3 utils.py"""


if __name__ == "__main__":
    df = load_training_samples()