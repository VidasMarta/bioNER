from datetime import datetime
import logging
import re
import string
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import argparse
import pandas as pd
import subprocess
from collections import defaultdict  
import sys
import requests
import ast


SPACY_NLP = None


def safe_load_response(response):
    raw_data = json.loads(response.content.decode("utf-8"))
    return normalize_llama_response(raw_data)


def normalize_llama_response(data):
    # Case 1: single dict → wrap in list
    if isinstance(data, dict):
        return [data]

    # Case 2: list
    if isinstance(data, list):
        normalized = []
        for item in data:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"content": item})
            else:
                normalized.append({"content": str(item)})
        return normalized

    # Case 3: raw string
    if isinstance(data, str):
        return [{"content": data}]

    # fallback
    return [{"content": str(data)}]


def remove_code_fences(text: str) -> str:
    return re.sub(r'```[\w]*\n?', '', text).strip()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = re.sub(r'```.*?```', '', text, flags=re.S)
    text = re.sub(r'^\s*json\s*', '', text, flags=re.I)
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'\*\*', '', text)
    text = text.strip()

    return text


def normalize_llm_json(text: str) -> str:
    text = clean_text(text)

    # convert: [ "x" ] → ["x"]
    text = re.sub(r'\[\s*"(.*?)"\s*\]', r'["\1"]', text)

    return text


def safe_parse(text: str):
    text = normalize_llm_json(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try Python literal lists with single quotes
    try:
        return ast.literal_eval(text)

    except Exception:
        print(f"[WARNING] Parsing failed for text: {text}")
        return []


def normalize_leading_cap(text: str) -> str:
    """
    Lowercase first letter only when the initial capital appears to be
    sentence-style capitalization rather than an acronym/code/token.

    Rules:
    - Empty / 1-char strings handled safely
    - If first char is not A-Z -> unchanged
    - If second char is missing -> lowercase single capital letter
    - Keep unchanged if second char is:
        - uppercase letter
        - digit
        - non-letter symbol
      Examples: USA, X1, A_Brand, C++
    - Keep unchanged if the whole alpha token is uppercase
      Example: NASA
    - Otherwise lowercase first letter
      Examples: Apple -> apple, Zagreb -> zagreb
    """
    if not text:
        return text

    if len(text) == 1:
        return text.lower() if text.isalpha() else text

    first = text[0]
    second = text[1]

    # First char must be uppercase letter
    if not first.isalpha() or not first.isupper():
        return text

    # Preserve if second char is uppercase, digit, or symbol
    if second.isupper() or second.isdigit() or not second.isalpha():
        return text

    # Extract leading alphabetic token
    m = re.match(r'^([A-Za-z]+)', text)
    if m:
        token = m.group(1)
        # Preserve all-caps words
        if token.isupper():
            return text

    # Lowercase only first character
    return first.lower() + text[1:]

def check_last_token(tokens_lower: List[str]) -> List[str]:
    tokens_lower[-1] = tokens_lower[-1].rstrip(string.punctuation)
    return tokens_lower


def create_concept_txt_file(file_path = '/home/${USERNAME}/syn-bioner/data/SNOMEDCT/CONCEPT.csv',
                            output_path = '/home/${USERNAME}/syn-bioner/data/SNOMEDCT/concepts.txt'):
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
    
def setup_logger(output_directory, verbose):
    log_dir = output_directory
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(log_dir, f"{date_str}_tags_generation.log")
    logger = logging.getLogger("tags_generation")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(fh)
    logger.propagate = False
    return logger
        
"""python3 utils.py"""


# TODO: move to utils and edit for DO, SNOMED CT and DO pairs and triplets.
# Structure so multiple languages can be read.
def generate_term_list(disease_file: str,
                       verbose: bool = True,
                       ) -> List[Tuple[str]]:
    entities = []
    data = []
    # Read the file (one dict per line)
    if disease_file.endswith('.json') or disease_file.endswith('.jsonl'):
        with open(disease_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
        term_list = [(d['disease_names'], d['doid']) for d in data]

    elif disease_file.endswith('.txt'):
        with open(disease_file, "r") as f:
            ents = f.readlines()
            ents = [str(ent).strip() for ent in ents]
        for ent in ents:
            entities.append({"idx": 0, "entity": ent})
        term_list = [(term['entity'],'') for term in entities]
        term_list = list(set(term_list))    
        term_list = [([t[0]],[t[1]]) for t in term_list]
    elif disease_file.endswith('.csv'):
        df = pd.read_csv(disease_file)
        term_list = [(str(row.iloc[0]).strip(), str(row.iloc[1]).strip()) for _, row in df.iterrows()]
        term_list = list(set(term_list))    
        term_list = [([t[0]],[t[1]]) for t in term_list]
              
    if verbose:
        print('Term_sample: ', term_list[:10]) 
    return term_list


if __name__ == "__main__":
    df = load_training_samples()