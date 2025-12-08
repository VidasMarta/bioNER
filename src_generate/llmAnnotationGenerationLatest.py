import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import tqdm
import os
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict  
import obonet
from src_generate import promptGeneration
from src_generate import utils


def argparse_args():
    parser = argparse.ArgumentParser(description="LLM-based text Generator.")
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the input text file to use as example of diseases.')
    parser.add_argument('--input_file_type', type=str, required=True, 
                        help='Path to the input text file to use as example of diseases.')
    parser.add_argument('--input_directory', type=str, default='',
                        help='Path to the input directory containing ontology terms as example of diseases.')
    parser.add_argument('--output_directory', type=str, required=True, 
                        help='Directory where sample sentences are outputed using JSON format.')
    parser.add_argument('--server_url', type=str, default="http://127.0.0.1:8080", 
                        help='URL of the llama.cpp inference server (default: http://127.0.0.1:8080).')
    parser.add_argument('--num_sentences', type=int, default=1, 
                        help='Number of sentences to produce by LLM.')
    parser.add_argument('--temperature', type=float, default=0.2, 
                        help='Sampling temperature for the LLM. (default: 0.2).')
    parser.add_argument('--max_tokens', type=int, default=2000, 
                        help='Maximum number of tokens to generate in LLM response (default: 2000).')
    parser.add_argument('--verbose', action='store_true', 
                        help='If set, print detailed debug output including LLM responses.')
    parser.add_argument('--use_context', action='store_true', 
                        help='If set, include context of the ontology term')    
    parser.add_argument('--obo_file_path', type=str, default='', 
                        help='Directory where sample sentences are outputed using JSON format.')
    parser.add_argument('--reprocess', action='store_true', 
                        help="""If set, recalculate the generation for allready existing 
                        sentences using terms.""")
    parser.add_argument('--test', action='store_true', 
                        help="""If set, recalculate the generation for allready existing 
                        sentences using terms.""")
    parser.add_argument('--kshot_path', type=str, default='', 
                        help='Path to the file containing training examples for few-shot prompting.')
    parser.add_argument('--kshot_size', type=int, default=0, 
                        help='Number of examples to use for each prompt.')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for reproducibility.')
    parser.add_argument('--include_pos', action='store_true', 
                        help='Include POS tags in k-shot examples.')
    parser.add_argument('--include_dep', action='store_true', 
                        help='Include dependency tags in k-shot examples.')
    parser.add_argument('--no_entity_ratio', type=float, default=0.25,
                        help='Ratio of sentences without entities.')


    return parser.parse_args()

def setup(args, iter_output):
    date_today = datetime.today().strftime("%Y%m%d")
    np.random.seed(getattr(args, "random_seed", 42))  # Fixed seed for reproducibility

    output_path = os.path.join(iter_output, f'generated_sentences_{date_today}.json')
    return output_path

def load_kshot_examples(args, kshot_path):
    kshot_examples = []
    if kshot_path and os.path.exists(kshot_path):
        kshot_examples = utils.load_training_samples(kshot_path)
        if args.verbose:
            print(f"[INFO] Loaded {len(kshot_examples)} k-shot examples from {kshot_path}")
    else:
        print(f"[INFO] No k-shot examples loaded. BUG! Check path: {kshot_path}")
        print(f"{os.path.exists(kshot_path)} {os.getcwd()}")

    return kshot_examples

def sample_kshot(args, no_entity_examples, entity_examples, kshot):
    if kshot:
        # Choose whether this sentence should contain an entity
        want_no_entity = np.random.rand() < args.no_entity_ratio

        if want_no_entity:
                            # Prefer no-entity examples
            if len(no_entity_examples) > 0:
                pool = no_entity_examples
                user_template = 'kshot_genre_no_entity'
            else:
                # fallback
                pool = entity_examples
                user_template = 'kshot_num_sent_genre_entity'
        else:
            # Prefer entity examples
            if len(entity_examples) > 0:
                pool = entity_examples
                user_template = 'kshot_num_sent_genre_entity'
            else:
                # fallback
                pool = no_entity_examples
                user_template = 'kshot_genre_no_entity'

    # FINAL fallback if both empty (should not happen)
    if len(pool) == 0:
        kshot_text_block = ""
        used_ids = []
    else:
        kshot_text_block, used_ids = utils.sample_k_examples(args, pool)

    return kshot_text_block, user_template, used_ids


def save_generated_sentences(args, output_path, method, response, term, used_ids):
    text = response.json()['content'].strip()
    text = utils.clean_text(text)
    text = utils.remove_code_fences(text)
    text, entities = utils.parse_text_entities_format(args, text)
    if args.verbose:
        args.logger.info(f"Generated text is: {text}")
        args.logger.info(
                f"Extracted entities proposed by LLM: {entities}\n Term is: {term}")
    record = {}
    with open(output_path, method, encoding='utf-8') as file:
        for t, ent in zip(text, entities):
            record["text"] = t
            record["entity"] = ent
            record["term"] = term
            record["kshot_example_ids"] = used_ids
            record["include_pos"] = getattr(args, "include_pos", True)
            record["include_dep"] = getattr(args, "include_dep", True)
            record["random_seed"] = getattr(args, "random_seed", 42)
            file.write(json.dumps(record))
            file.write("\n")


def generate_sentences_per_cluster(
    args: argparse.Namespace,
    iter_output: str, 
    kshot_path: str,
    term_list: List[str], 
    clusters: List[int],
    num_of_terms_pc: List[int],
    method: str = 'a',
    system_template: str = 'role_prompt',
    user_template: str = 'genre_prompt'
) -> str:
    output_path = setup(args)
    kshot_examples = load_kshot_examples(args, kshot_path)

    start = 0 
    for i, cluster in enumerate(clusters):
        try:
            kshot_pool = [ex for ex in kshot_examples if ex.get("cluster_id") == cluster]
            entity_examples = [ex for ex in kshot_pool if ex.get("entities")]
            no_entity_examples = [ex for ex in kshot_pool if not ex.get("entities")]

            end = start + num_of_terms_pc[i]
            terms = term_list[start:end]
            start = end

            for term in terms:
                term = np.random.choice(term_list, replace=False) #take a random term
                kshot_text_block, user_template, used_ids = sample_kshot(args, no_entity_examples, entity_examples, True)
                args.logger.info(f"K-shot examples used for term '{term}': {used_ids}")

                response = promptGeneration.message_request(
                        args,
                        term,
                        system_template=system_template,
                        user_template=user_template,
                        text=kshot_text_block
                    )
                
                save_generated_sentences(args, output_path, method, response, term, used_ids)
        except Exception as e:
                args.logger.info(f"Failed to generate or parse sentence for cluster {cluster}: {e}")
                args.logger.info(f"Response content: {response.json().get('content', '')}")
                if args.verbose:
                    print(f"[ERROR] Term {term}: {e}")
    return output_path

    
def generate_sentence_samples(
    args: argparse.Namespace,
    kshot_path: str,
    term_list: List[str], 
    method: str = 'a',
    system_template: str = 'role_prompt',
    user_template: str = 'genre_prompt'
)-> str:
    """
    Generate sentences for a list of disease terms using optional k-shot examples.
    - Randomly samples k-shot examples (from given NCBI subset) per term with fixed seed for reproducibility.
    - Allows toggling inclusion of POS/DEP features in few-shot examples.
    - Stores which k-shot example IDs were used in each generated JSON output.

    returns path to the output JSON file.
    """
    output_path = setup(args)
    kshot_examples = load_kshot_examples(args, kshot_path)
    
    # Filter by havig entity or not in a kshot_examples pool
    entity_examples = [ex for ex in kshot_examples if ex.get("entities")]
    no_entity_examples = [ex for ex in kshot_examples if not ex.get("entities")]

    for i, term in enumerate(tqdm.tqdm(term_list)):
        with open(output_path, method, encoding='utf-8') as file:
            try:
                if kshot_examples:
                    kshot = True
                else:
                    kshot = False
                kshot_text_block, user_template, used_ids = sample_kshot(args, no_entity_examples, entity_examples, kshot)

                args.logger.info(f"K-shot examples used for term '{term}': {used_ids}")

                response = promptGeneration.message_request(
                        args,
                        term,
                        system_template=system_template,
                        user_template=user_template,
                        text=kshot_text_block
                    )
                
                save_generated_sentences(args, output_path, method, response, term, used_ids)  

            except Exception as e:
                args.logger.info(f"Failed to generate or parse sentence for term {term}: {e}")
                args.logger.info(f"Response content: {response.json().get('content', '')}")
                if args.verbose:
                    print(f"[ERROR] Term {term}: {e}")
    return output_path

def get_diseases(args: argparse.Namespace):
    graph = obonet.read_obo(args.obo_file_path)
    do_terms = [(data["name"], node) for node, data in graph.nodes(data=True) if "name" in data]
    if args.verbose:
        print(f"Number of nodes (terms): {graph.number_of_nodes()}")
        print(f"Number of edges (relations): {graph.number_of_edges()}")
        print('First 20 terms: ', do_terms[:20])  # Show first 20 terms
    return do_terms 

def generate_term_list(args: argparse.Namespace):
    entities = []
    # Read the file (one dict per line)
    if args.input_file_type.lower() == 'json':
        with open(args.input_file, "r") as f:
            data = [json.loads(line.strip().replace("'", '"')) for line in f if line.strip()]
        grouped = defaultdict(list)
        for item in data:
            grouped[item['idx']].append(item)

        # Process each idx
        for idx, items in grouped.items():
            tokens = []
            capture = False
            for item in items:
                gold = item['gold']
                if gold.startswith("B-"):
                    if tokens:  # flush previous entity
                        entities.append({"idx": idx, "entity": " ".join(tokens)})
                        tokens = []
                    tokens.append(item['token'])
                    capture = True
                elif gold.startswith("I-") and capture:
                    tokens.append(item['token'])
                else:
                    if tokens:  # flush if ended
                        entities.append({"idx": idx, "entity": " ".join(tokens)})
                        tokens = []
                    capture = False

            if tokens:  # flush last
                entities.append({"idx": idx, "entity": " ".join(tokens)})
        term_list = [(term['entity'],'NaN') for term in entities]

    elif args.input_file.endswith('.txt'):
        with open(args.input_file, "r") as f:
            ents = f.readlines()
            ents = [str(ent).strip() for ent in ents]
        for ent in ents:
            entities.append({"idx": 0, "entity": ent})
        term_list = [(term['entity'],'NaN') for term in entities]
        
    elif args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
        term_list = [(str(row.iloc[0]).strip(), str(row.iloc[1]).strip()) for _, row in df.iterrows()]
        
    term_list = list(set(term_list))    
    
    return term_list   
    
    
def main(args: argparse.Namespace):
    os.makedirs(args.output_directory, exist_ok=True)
    # open json file where each line is one dict
    term_list = generate_term_list(args)
    
    if args.obo_file_path:
        disease_terms = utils.get_diseases(args)
        term_list += disease_terms
    # print(term_list[:150])
    if args.test:
        term_list = term_list[:2] + term_list[400:406] + term_list[1100:1102] + term_list[-2:]
        print("Testing on samples: ", len(term_list), term_list)

    generate_sentence_samples(args, args.kshot_path, term_list)
    
def setup_logger(args):
    log_dir = args.output_directory
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(log_dir, f"{date_str}_tags_generation.log")
    logger = logging.getLogger("tags_generation")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(fh)
    logger.propagate = False
    return logger

"""
With included pos and dep
python3 llmAnnotationGeneration.py \
  --input_file data/NCBI-Disease/test.txt \
  --input_file_type list \
  --output_directory data/synthetic_aug \
  --server_url http://0.0.0.0:8484 \
  --kshot_path data/ncbi_ner_train_10pct.json \
  --kshot_size 5 \
  --num_sentences 2 \
  --include_pos \
  --include_dep \
  --random_seed 42 \
  --verbose


python3 bioNER/src-generate/llmAnnotationGeneration.py \
  --input_file data/SNOMEDCT/concepts_filtered.csv \
  --input_file_type list \
  --output_directory  data/synthetic-snomed-kshot \
  --server_url http://med-llm-webapp-backend-1:8080  \
  --kshot_path data/ncbi/trf/ncbi_ner_train_10pct.json \
  --kshot_size 5 \
  --num_sentences 2 \
  --include_pos \
  --include_dep \
  --random_seed 42 \
  --verbose \
  --test --temperature 0 --max_tokens 500 --verbose
  
  
  Excluded pos and dep
  python3 llmAnnotationGeneration.py \
  --input_file data/NCBI-Disease/test.txt \
  --output_directory data/synthetic_no_tags \
  --kshot_path data/ncbi_ner_train_10pct.json \
  --kshot_size 5 \
  --num_sentences 2 \
  --random_seed 42

"""
if __name__ == "__main__":
    args = argparse_args()
    args.logger = setup_logger(args)
    args.logger.info(f"Output directory: {args.output_directory}")
    main(args)