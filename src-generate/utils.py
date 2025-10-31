import re
import string

import numpy as np
import obonet
from typing import List, Dict, Any, Optional
import argparse
import pandas as pd
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


#TODO: LOAD JSON TRAINING SAMPLES      
# /home/mkeber/syn-bioner/data/NCBI-Disease/5_selected_ncbi_sentences.csv
# /home/mkeber/syn-bioner/data/NCBI-Disease/merged_features_3.json
# RETURN FOR EACH CLUSTER KEY IN DICTIONARY A LIST OF TUPLES (SENTENCE, DISEASES), i.e. (STR, LIST-OF-STRINGS)
def load_training_samples(
    file_path: str
    ) -> List[Dict[str, Any]]:
    df = pd.read_json(file_path)
    print(f"Loaded {len(df)} samples from {file_path}")

    return df

def format_kshot_block(examples: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    formatted = []
    for ex in examples:
        ex_text = ex.get("sentence", "").strip()
        entities = ", ".join(ex.get("entities", []))
        block = f"Sentence: {ex_text}\nEntities: [{entities}]"

        # Conditionally add linguistic features
        if getattr(args, "include_pos", True):
            pos_tags = " ".join(ex.get("pos", []))
            block += f"\nPOS: {pos_tags}"
        if getattr(args, "include_dep", True):
            deps = " ".join(ex.get("dep", []))
            block += f"\nDEP: {deps}"

        formatted.append(block + "\n")
    return "\n".join(formatted)

def sample_k_examples(args: argparse.Namespace, kshot_pool):
    k = min(args.kshot_size, len(kshot_pool))
    sampled = np.random.choice(kshot_pool, size=k, replace=False)
    kshot_text_block = format_kshot_block(sampled, args)
    used_ids = [ex.get("id", f"ex_{idx}") for idx, ex in enumerate(sampled)]

    return kshot_text_block, used_ids



"""python3 utils.py"""


if __name__ == "__main__":
    df = load_training_samples()
    print(df.entities.notna())
    print(df.entities)