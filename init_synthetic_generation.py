import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import tqdm
import os
import random
import numpy as np
from datetime import datetime
from collections import defaultdict

import yaml  
from src_generate import prompt_generation
from src_generate import utils
import generation_postprocessing
import obonet

def argparse_args():
    parser = argparse.ArgumentParser(description="LLM-based text Generator iteration pipeline with k-shot.")
    parser.add_argument('--config_file', type=str, default=f'/home/mkeber/syn-bioner/bioNER/experiments/init_generation.yml', help='Path to config file with all arguments.')

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


def save_generated_sentences(args, output_path, method, response, term, 
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
        file.write(json.dumps(record))
        file.write("\n")
    return text

    
def generate_sentence_samples(
    args: argparse.Namespace,
    term_list: List[tuple], 
    method: str = 'a',
    system_template: str = 'role_prompt',
    user_template: str = 'genre_prompt',
    nlp: Any = None
)-> str:
    """
    Generate sentences for a list of disease terms using optional k-shot examples.
    - Randomly samples k-shot examples (from given NCBI subset) per term with fixed seed for reproducibility.
    - Allows toggling inclusion of POS/DEP features in few-shot examples.
    - Stores which k-shot example IDs were used in each generated JSON output.

    returns path to the output JSON file.
    """
    output_path, corrected_output_path = setup(args)
    for i, term in enumerate(tqdm.tqdm(term_list)):
        try:
            args.logger.info(f"Term '{term[0][0]}'!")
            if len(term[0]) > 1:
                text_term = "and ".join(term[0])
            else: text_term = term[0]            
            response = prompt_generation.message_request(
                        args,
                        text_term,
                        system_template=system_template,
                        user_template=user_template
                )
            text = save_generated_sentences(args, output_path, method, response, term)
            generated_json = generation_postprocessing.create_json(args, text, i, term, nlp)
            with open(corrected_output_path, method, encoding='utf-8') as file:
                file.write(json.dumps(generated_json))
                file.write("\n")
            
        except Exception as e:
            args.logger.info(f"Failed to generate or parse sentence for term {term[0]}: {e}")
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
    # TODO: pairs and triplets of terms.
    if args.pairs_file_path:
        disease_terms = utils.generate_term_list(args.pairs_file_path, args.verbose)
        term_list += disease_terms
    # print(term_list[:150])
    if args.test:
        term_list = term_list[:2] + term_list[400:406] + term_list[1100:1102] + term_list[-2:]
        print("Testing on samples: ", len(term_list), term_list)
    random.seed(args.random_seed)
    term_list = random.sample(term_list, args.generate_k)

    nlp = generation_postprocessing.spacy_load_model('en_core_web_trf')
    # TODO: system_template, user_template to args
    generate_sentence_samples(args, term_list, 
                              system_template='role_sent_type_prompt',
                              user_template='genre_new_prompt',
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
