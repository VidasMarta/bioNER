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
import traceback
import yaml  
from .src_generate import prompt_generation
from .src_generate import utils
from . import generation_postprocessing
import obonet
import time

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


def save_generated_sentences(args, output_path, method, text, term, sent_gen_time,
                             used_ids: list = []):
    # text = response.json()['content'].strip()
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
        filtered = [
            item for item in kshot_examples
            if item.get("sentence", "").strip()                      # not empty
            and not item.get("sentence", "").isupper()              # not ALL CAPS
            and len(item.get("parents", [])) >= 3                   # parents length at least 3
        ]
        return filtered

# result is in filtered
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
    args.output_path, args.corrected_output_path = setup(args, iter_output)
    args.logger.info(f"[INFO] Output path: {args.output_path}")
    args.logger.info(f"[INFO] Corrected output path: {args.corrected_output_path}")
    # print(f"[INFO] Output path: {output_path}",
    #       f"\n[INFO] Corrected output path: {corrected_output_path}")
    kshot_pool = load_kshot_examples(args, args.kshot_pool)
    seen = set()

    for i, ex in enumerate(kshot_pool):
        ex_id = ex.get('id')

        if ex_id is None or ex_id in seen:
            ex_id = i
            while ex_id in seen:
                ex_id += 1
            ex['id'] = ex_id

        seen.add(ex_id)
            
    # open(output_path, "w").close()  
    batch_size = max(1, args.batch)

    # Prepare items first
    items = []
    for i, term in enumerate(term_list):
        kshot_text_block, user_template, used_ids = sample_kshot(args, kshot_pool)
        for i, t in enumerate(term[0]):
            term[0][i] = utils.normalize_leading_cap(t)
        # print(used_ids)
        items.append({
            "term": term,
            "term_text": term[0],
            "text_block": kshot_text_block,
            "user_template": user_template,
            "system_template": system_template,
            "used_ids": used_ids,
            "index": i
        })
    for i in tqdm.tqdm(
        range(0, len(items), batch_size),
        desc="Processing batched k-shot generation",
        total=(len(items) + batch_size - 1) // batch_size
    ):
        batch = items[i:i+batch_size]
        batch_timings = {}
        batch_timings['start_time'] = time.time()
#========================================
#       k shot prompt 
#========================================
        try:  
            response = prompt_generation.message_request(
                    args,
                    batch
                )
            batch_gen_time = time.time()
            batch_timings['batch_gen_time'] = batch_gen_time - batch_timings['start_time']
            args.timings.append(batch_timings)
            data = utils.safe_load_response(response)
            for d in data:
                args.logger.info(f'RESPONSE {i} for generation:  {d["content"]}')
            texts = [save_generated_sentences(args, args.output_path, 'a', d['content'].strip(), 
                            item['term'], batch_timings['batch_gen_time'], item['used_ids']) for item, d in zip(batch,data)]
            print("saved to ", args.output_path)
            terms = [item['term'] for item in batch]
            kshot_text_blocks = [item['text_block'] for item in batch]
#========================================
#       Computation of correction 
#========================================
            # TODO: this is completely fixed should we add kshot.
            generated_jsons = generation_postprocessing.create_json(args, texts, i, terms, nlp, 
                                            system_template='annotation', 
                                            user_template='disease_annotation_reduced', 
                                            kshot_text_blocks=kshot_text_blocks)
            
            args.timings[-1]['batch_correction_time'] = time.time() - batch_gen_time
            # print("created json: ", generated_jsons)
            args.timings[-1]['total_sample_time'] = time.time() - batch_timings['start_time']

#========================================
#       Save outputs
#========================================
            with open(args.corrected_output_path, 'a', encoding='utf-8') as file:
                for generated_json in generated_jsons:
                    file.write(json.dumps(generated_json))
                    file.write("\n")
            # print("saved corrected to ", corrected_output_path)
            with open(os.path.join(args.output_directory, 'term_list.txt'), 'a') as f:
                for term in terms:
                    f.write(str(term) + '\n')
                                
            with open(os.path.join(args.output_directory, 'timings.json'), 'a') as fi:
                fi.write(json.dumps(args.timings[-1]))
                fi.write("\n")

        except Exception as e:
            args.logger.info(f"[ERROR]Response content: {data}")
            if args.verbose:
                print(f"[ERROR] Term {term[0]}")
                traceback.print_exc()
    return args.output_path


def how_many_to_generate(args):
    for file in os.listdir(args.output_directory):
        if 'corrected_generated_sentences_' in file:
            with open(file, "r", encoding="utf-8") as f:
                num_lines = sum(1 for _ in f)
            if num_lines < args.generate_k:
                args.generate_k -= num_lines


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
    # fill up the output or use the 
    how_many_to_generate(args)
    if args.test: 
        args.generate_k = 100
        # print("Testing on samples: ", len(term_list), term_list)
    random.seed(args.random_seed)

    if len(term_list) > args.generate_k:
        term_list = random.sample(term_list, args.generate_k)
    else:
        reps = random.sample(term_list,args.generate_k - len(term_list))
        term_list += reps

    nlp = generation_postprocessing.spacy_load_model(args.spacy_model)
    # TODO: system_template, user_template to args
    # It is defined within kshot sampling depends on 
    # the ration of no entitiy sentences
    kshot_generation(args, 
                     '', 
                     term_list, 
                     system_template=args.system_template,
                    nlp = nlp)


def load_config(config_path: str):
    import yaml, os

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    def expand(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        return value

    config = {k: expand(v) for k, v in config.items()}

    # ✅ cast types explicitly
    int_fields = ["k_shot", "batch", "num_sentences", "max_tokens",
                  "generate_k", "random_seed", "kshot_size", ]
    
    float_fields = ["temperature", "no_entity_ratio"]

    bool_fields = ["verbose", "use_context", "test", "reprocess", 
                   "include_pos", "include_dep", "obo_file_path"]

    for key in int_fields:
        if key in config and config[key] is not None:
            config[key] = int(config[key])

    for key in float_fields:
        if key in config and config[key] is not None:
            config[key] = float(config[key])

    for key in bool_fields:
        if key in config and isinstance(config[key], str):
            config[key] = config[key].lower() in ("true", "1", "yes")

    return argparse.Namespace(**config)


if __name__ == "__main__":
    init_args = argparse_args()
    args = load_config(init_args.config_file)
    args.logger = utils.setup_logger(args.output_directory, args.verbose)
    args.timings = []
    vars_str = '{'
    for k, v in vars(args).items():
        vars_str += f'\n {k}: {v},'
    vars_str += '}'

    args.logger.info(f"Arguments:\n {vars_str}")
    args.logger.info(f"Output directory: {args.output_directory}")
    
    main(args)
