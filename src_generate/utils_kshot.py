import json
import argparse
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import tqdm
import os
import numpy as np
from datetime import datetime
from collections import defaultdict  
import obonet
from src_generate import prompt_generation
from src_generate import utils

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

def sample_kshot(args, no_entity_examples, entity_examples):
    # Choose whether this sentence should contain an entity
    want_no_entity = np.random.rand() < args.no_entity_ratio

    if want_no_entity:
                            # Prefer no-entity examples
        if len(no_entity_examples) > 0:
            pool = no_entity_examples
            user_template = 'kshot_no_entity'
        else:
            # fallback
            pool = entity_examples
            user_template = 'kshot_entity'
    else:
        # Prefer entity examples
        if len(entity_examples) > 0:
            pool = entity_examples
            user_template = 'kshot_entity'
        else:
            # fallback
            pool = no_entity_examples
            user_template = 'kshot_no_entity'

    # FINAL fallback if both empty (should not happen)
    print(f"[INFO] wnat no entity: {want_no_entity}")
    print(f"[INFO] pool: {len(pool)}")
    if len(pool) == 0:
        kshot_text_block = ""
        used_ids = []
        print("[DEBUG] This shouldn't be happening...")
    else:
        kshot_text_block, used_ids = utils.sample_k_examples(args, pool)

    return kshot_text_block, user_template, used_ids


def save_generated_sentences(args, output_path, response, term, used_ids, sent_gen_time):
    text = response.json()['content'].strip()
    text = utils.clean_text(text)
    text = utils.remove_code_fences(text)
    #parsed_text, entities = utils.parse_text_entities_format(args, text)
    if args.verbose:
        args.logger.info(f"Generated text is: {text}")
        args.logger.info(f"Term is: {term}")
                #f"Extracted entities proposed by LLM: {entities}\n Term is: {term}")
    record = {}
    with open(output_path, 'a', encoding='utf-8') as file:
        record["text"] = text
        record["entity"] = []
        record["term"] = term
        record["kshot_example_ids"] = used_ids
        record["include_pos"] = getattr(args, "include_pos", True)
        record["include_dep"] = getattr(args, "include_dep", True)
        record["random_seed"] = getattr(args, "random_seed", 42)
        record["time"] = sent_gen_time
        file.write(json.dumps(record))
        file.write("\n")


def generate_sentences_per_cluster(
    args: argparse.Namespace,
    iter_output: str, 
    kshot_path: str,
    term_list: List[tuple], 
    clusters: List[int],
    num_of_terms_pc: List[int], #number of terms per cluster
    system_template: str = 'role_prompt'
) -> str:
    output_path = setup(args, iter_output)
    kshot_examples = load_kshot_examples(args, kshot_path)
    open(output_path, "w").close()  

    start = 0 
    print("Total terms:", len(term_list))
    print("Sum per cluster:", sum(num_of_terms_pc))

    for i, cluster in enumerate(tqdm.tqdm(clusters)):
        try:
            kshot_pool = [ex for ex in kshot_examples if ex.get("cluster_id") == cluster]
            print(kshot_pool[:4])
            entity_examples = [ex for ex in kshot_pool if ex.get("entities")]
            no_entity_examples = [ex for ex in kshot_pool if not ex.get("entities")]

            end = start + num_of_terms_pc[i]
            terms = term_list[start:end]
            start = end

            if not terms:
                print(f"[WARNING] No terms for cluster {cluster}")
                continue

            for term in terms:
                start_time = time.time()
                print(f"[INFO] Generating sentence for term: {term[0]}")
                kshot_text_block, user_template, used_ids = sample_kshot(args, no_entity_examples, entity_examples)
                args.logger.info(f"K-shot examples used for term '{term[0]}': {used_ids}")
                #TODO:
                # define a batch
                batch=[{}]
                response = prompt_generation.message_request(
                        args,
                        batch,
                        system_template=system_template,
                        user_template=user_template,
                    )
                sent_gen_time = time.time() - start_time
                save_generated_sentences(args, output_path, response, term, used_ids, sent_gen_time)
        except Exception as e:
                args.logger.info(f"Failed to generate or parse sentence for cluster {cluster}: {e}")
                args.logger.info(f"Response content: {response.json().get('content', '')}")
                if args.verbose:
                    print(f"[ERROR] Term {term[0]}: {e}")

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
    if args.disease_file_type.lower() == 'json':
        with open(args.disease_file, "r") as f:
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

    elif args.disease_file.endswith('.txt'):
        with open(args.disease_file, "r") as f:
            ents = f.readlines()
            ents = [str(ent).strip() for ent in ents]
        for ent in ents:
            entities.append({"idx": 0, "entity": ent})
        term_list = [(term['entity'],'NaN') for term in entities]
        
    elif args.disease_file.endswith('.csv'):
        df = pd.read_csv(args.disease_file)
        term_list = [(str(row.iloc[0]).strip(), str(row.iloc[1]).strip()) for _, row in df.iterrows()]
        
    term_list = list(set(term_list))    
    
    return term_list   
    
    
def main(args: argparse.Namespace):
    os.makedirs(args.output_directory, exist_ok=True)
    # open json file where each line is one dict
    term_list = generate_term_list(args)
    
    if args.obo_file_path:
        disease_terms = get_diseases(args)
        term_list += disease_terms
    # print(term_list[:150])
    if args.test:
        term_list = term_list[:2] + term_list[400:406] + term_list[1100:1102] + term_list[-2:]
        print("Testing on samples: ", len(term_list), term_list)

    
if __name__ == "__main__":
    args = argparse_args()
    args.logger = utils.setup_logger(args)
    args.logger.info(f"Output directory: {args.output_directory}")
    main(args)