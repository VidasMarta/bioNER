import json
import argparse
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import tqdm
import os
import random
import numpy as np
from datetime import datetime
from collections import defaultdict

import yaml  
from .src_generate import prompt_generation
from .src_generate import utils
from . import generation_postprocessing
import obonet

def argparse_args():
    parser = argparse.ArgumentParser(
        description="LLM-based text Generator iteration pipeline with k-shot.")
    parser.add_argument('--config_file', type=str, 
            default=f'/home/mkeber/syn-bioner/bioNER/experiments/kshot_generation.yml', 
            help='Path to config file with all arguments.')

    return parser.parse_args()

def setup(args: argparse.Namespace, iter_output: str = '') -> str:
    date_today = datetime.today().strftime("%Y%m%d")
    np.random.seed(getattr(args, "random_seed", 42))  # Fixed seed for reproducibility
    if iter_output: 
        out_dir = iter_output
    else: 
        out_dir = args.output_directory
    output_path = os.path.join(out_dir, f'generated_sentences_{date_today}.jsonl')
    corrected_output_path = os.path.join(out_dir, f'corrected_generated_sentences_{date_today}.jsonl')
    return output_path, corrected_output_path


def save_generated_sentences(args, output_path, method, response, term, sent_gen_time,
                             used_ids: list = []):
    text = response.json()['content'].strip()
    text = utils.clean_text(text)
    text = utils.remove_code_fences(text)
    #parsed_text, entities = utils.parse_text_entities_format(args, text)
    if args.verbose:
        args.logger.info(f"Generated text is: {text}")
        args.logger.info(f"Term is: {term}")
                #f"Extracted entities proposed by LLM: {entities}\n Term is: {term}")
    record = {}
    with open(output_path, method, encoding='utf-8') as file:
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
    return text

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


def sample_kshot(args, kshot_pool) -> Tuple[str, str, list]:
    # Choose whether this sentence should contain an entity
    if not kshot_pool:
        return '', args.user_template, [] # kshot_text_block, user_template, used_ids
    want_no_entity = np.random.rand() < args.no_entity_ratio
    entity_examples = [ex for ex in kshot_pool if ex.get("entities")]
    no_entity_examples = [ex for ex in kshot_pool if not ex.get("entities")]

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
    args.logger.info(f"[INFO] want no entity: {want_no_entity}")
    args.logger.info(f"[INFO] pool: {len(pool)}")
    if len(pool) == 0:
        kshot_text_block = ""
        used_ids = []
        args.logger.info("[DEBUG] This shouldn't be happening...")
    else:
        kshot_text_block, used_ids = utils.sample_k_examples(args, pool)

    return kshot_text_block, user_template, used_ids

def kshot_generation(
    args: argparse.Namespace,
    iter_output: str, 
    term_list: List[tuple], 
    system_template: str = 'role_prompt',
    nlp: Any = None
    ) -> str:
    output_path, corrected_output_path = setup(args, iter_output)
    args.logger.info(f"[INFO] Output path: {output_path}")
    args.logger.info(f"[INFO] Corrected output path: {corrected_output_path}")
    # print(f"[INFO] Output path: {output_path}",
    #       f"\n[INFO] Corrected output path: {corrected_output_path}")
    kshot_pool = load_kshot_examples(args, args.kshot_pool)
    # open(output_path, "w").close()  

    for i, term in enumerate(tqdm.tqdm(term_list)):
        try:
            start_time = time.time()
            print(f"[INFO] Generating sentence for term: {term[0]}")
            kshot_text_block, user_template, used_ids = sample_kshot(args, kshot_pool)
            args.logger.info(f"K-shot examples used for term '{term[0]}': {used_ids}")
            
            response = prompt_generation.message_request(
                    args,
                    term[0],
                    system_template=system_template,
                    user_template=user_template,
                    text=kshot_text_block
                )
            sent_gen_time = time.time() - start_time
            text = save_generated_sentences(args, output_path, 'a', response, term, sent_gen_time, used_ids)
            print("saved to ", output_path)
            generated_json = generation_postprocessing.create_json(args, text, i, term, nlp, 
                                            user_template='disease_annotation_reduced')
            print("created json: ", generated_json)
            with open(corrected_output_path, 'a', encoding='utf-8') as file:
                file.write(json.dumps(generated_json))
                file.write("\n")
            print("saved corrected to ", corrected_output_path)
            with open(os.path.join(args.output_directory, 'term_list.txt'), 'a') as f:
                f.write(str(term) + '\n')

        except Exception as e:
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

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_directory, exist_ok=True)
    # open json file where each line is one dict
    term_list = utils.generate_term_list(args.disease_file, args.verbose)
    if os.path.isfile(os.path.join(args.output_directory, 'term_list.txt')):
        with open(os.path.join(args.output_directory, 'term_list.txt'), 'r') as f:
            used_terms = [line.strip() for line in f.readlines()]
        # term_list = list(set(term_list) - set(used_terms))
    if args.obo_file_path:
        disease_terms = get_diseases(args)
        term_list += disease_terms
    # TODO: pairs and triplets of terms.
    if args.pairs_file_path:
        disease_terms = utils.generate_term_list(args.pairs_file_path, args.verbose)
        term_list += disease_terms

    if args.test:
        term_list = term_list[:2] + term_list[400:406] + term_list[1100:1102] + term_list[-2:]
        print("Testing on samples: ", len(term_list), term_list)
    random.seed(args.random_seed)

    if len(term_list) > args.generate_k:
        term_list = random.sample(term_list, args.generate_k)
    else:
        reps = random.sample(term_list,args.generate_k - len(term_list))
        term_list += reps
        
    nlp = generation_postprocessing.spacy_load_model(args.spacy_model)
    # TODO: system_template, user_template to args
    # Defined later depends on the ration of no entitiy sentences
    kshot_generation(args, '', term_list, 
                              system_template=args.system_template,
                              nlp = nlp)


if __name__ == "__main__":
    init_args = argparse_args()
    with open(init_args.config_file, 'r') as file:
        yaml_args = yaml.safe_load(file)
    args = argparse.Namespace(**yaml_args)
    args.logger = utils.setup_logger(args.output_directory, args.verbose)
    vars_str = '{'
    for k, v in vars(args).items():
        vars_str += f'\n {k}: {v},'
    vars_str = '}'

    args.logger.info(f"Arguments:\n {vars_str}")
    args.logger.info(f"Output directory: {args.output_directory}")
    main(args)
