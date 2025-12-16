import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import tqdm
import os
import numpy as np
from datetime import datetime
from collections import defaultdict  
from bioNER.src_generate import promptGeneration
from bioNER.src_generate import utils
from bioNER import generation_postprocessing


def argparse_args():
    parser = argparse.ArgumentParser(description="LLM-based text Generator.")
    parser.add_argument('--disease_file', type=str, default='', 
                        help='Path to the input text file to use as example of diseases.')
    parser.add_argument('--disease_file_type', type=str, default='', 
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
    parser.add_argument('--pairs_file_path', type=str, default='', 
                        help='Path to the input text file to use as example of diseases.')
    parser.add_argument('--iter_output')


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

    if want_no_entity and len(no_entity_examples) > 0:
        pool = no_entity_examples
        user_template = 'kshot_genre_no_entity'
    elif len(entity_examples) > 0:
        pool = entity_examples
        user_template = 'kshot_num_sent_genre_entity'
    elif len(no_entity_examples) > 0:
        # fallback
        pool = no_entity_examples
        user_template = 'kshot_genre_no_entity'
    else:
        # Fall back to basic generation without kshot
        pool = []
        user_template = 'genre_syn_generation_new'

    # FINAL fallback if both empty (should not happen)
    print(f"[INFO] wnat no entity: {want_no_entity}")
    print(f"[INFO] pool: {len(pool)}")
    if len(pool) == 0:
        kshot_text_block = ""
        used_ids = []
        print("[DEBUG] This shouldn't be happening... It is possible to be the basic generation pipeline.")
    else:
        kshot_text_block, used_ids = utils.sample_k_examples(args, pool)

    return kshot_text_block, user_template, used_ids


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

def generate_sentences_per_cluster(
    args: argparse.Namespace,
    iter_output: str, 
    kshot_path: str,
    term_list: List[tuple], 
    clusters: List[int],
    num_of_terms_pc: List[int], #number of terms per cluster
    method: str = 'a',
    system_template: str = 'role_prompt',
    user_template: str = 'genre_prompt'
) -> str:
    output_path, _ = setup(args, iter_output)
    kshot_examples = load_kshot_examples(args, kshot_path)
    open(output_path, "w").close()  

    start = 0 
    print("Total terms:", len(term_list))
    print("Sum per cluster:", sum(num_of_terms_pc))

    for i, cluster in enumerate(tqdm.tqdm(clusters)):
        try:
            kshot_pool = [ex for ex in kshot_examples if ex.get("cluster_id") == cluster]
            entity_examples = [ex for ex in kshot_pool if ex.get("entities")]
            no_entity_examples = [ex for ex in kshot_pool if not ex.get("entities")]

            end = start + num_of_terms_pc[i]
            terms = term_list[start:end]
            start = end

            if not terms:
                print(f"[WARNING] No terms for cluster {cluster}")
                continue

            for term in terms:
                print(f"[INFO] Generating sentence for term: {term[0]}")
                kshot_text_block, user_template, used_ids = sample_kshot(args, no_entity_examples, entity_examples)
                args.logger.info(f"K-shot examples used for term '{term[0]}': {used_ids}")

                response = promptGeneration.message_request(
                        args,
                        term[0],
                        system_template=system_template,
                        user_template=user_template,
                        text=kshot_text_block
                    )
                
                save_generated_sentences(args, output_path, method, response, term, used_ids)
        except Exception as e:
                args.logger.info(f"Failed to generate or parse sentence for cluster {cluster}: {e}")
                args.logger.info(f"Response content: {response.json().get('content', '')}")
                if args.verbose:
                    print(f"[ERROR] Term {term[0]}: {e}")

    return output_path

    
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
            response = promptGeneration.message_request(
                        args,
                        text_term,
                        system_template=system_template,
                        user_template=user_template
                )
            text = save_generated_sentences(args, output_path, method, response, term)  
            generated_json = generation_postprocessing.create_json(args, text, term, nlp, 
                                                                   user_template='disease_annotation_reduced')
            with open(corrected_output_path, method, encoding='utf-8') as file:
                file.write(json.dumps(generated_json))
                file.write("\n")
            
        except Exception as e:
            args.logger.info(f"Failed to generate or parse sentence for term {term[0]}: {e}")
            args.logger.info(f"Response content: {response.json().get('content', '')}")
            if args.verbose:
                print(f"[ERROR] Term {term[0]}: {e}")
    return output_path


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_directory, exist_ok=True)
    # open json file where each line is one dict
    term_list = utils.generate_term_list(args.disease_file, args.verbose)
    
    if args.obo_file_path:
        disease_terms = utils.get_diseases(args.obo_file_path, args.verbose)
        term_list += disease_terms
    # TODO: pairs and triplets of terms.
    if args.pairs_file_path:
        disease_terms = utils.generate_term_list(args.pairs_file_path, args.verbose)
        term_list += disease_terms
    # print(term_list[:150])
    if args.test:
        term_list = term_list[:2] + term_list[400:406] + term_list[1100:1102] + term_list[-2:]
        print("Testing on samples: ", len(term_list), term_list)
    nlp = generation_postprocessing.spacy_load_model('en_core_web_trf')
    generate_sentence_samples(args, term_list, 
                              system_template='role_sent_type_prompt',
                              user_template='genre_new_prompt',
                              nlp = nlp)


"""
With included pos and dep
python3 llmAnnotationGeneration.py \
  --disease_file data/NCBI-Disease/test.txt \
  --disease_file_type list \
  --output_directory data/synthetic_aug \
  --server_url http://0.0.0.0:8484 \
  --kshot_path data/ncbi_ner_train_10pct.json \
  --kshot_size 5 \
  --num_sentences 2 \
  --include_pos \
  --include_dep \
  --random_seed 42 \
  --verbose


python3 -m bioNER.src_generate.llmAnnotationGenerationLatest \
  --disease_file data/SNOMEDCT/concepts_filtered.csv \
  --disease_file_type list \
  --pairs_file_path data/hetionet/pairs.jsonl \
  --output_directory  data/synthetic_snomed_sent_type \
  --server_url http://172.19.0.2:8484  \
  --kshot_path data/ncbi/trf/ncbi_ner_train_10pct.json \
  --kshot_size 0 \
  --num_sentences 2 \
  --include_pos \
  --include_dep \
  --random_seed 42 \
  --test --temperature 0 --max_tokens 500 --verbose
  
  
  Excluded pos and dep
  python3 llmAnnotationGeneration.py \
  --disease_file data/NCBI-Disease/test.txt \
  --output_directory data/synthetic_no_tags \
  --kshot_path data/ncbi_ner_train_10pct.json \
  --kshot_size 5 \
  --num_sentences 2 \
  --random_seed 42

"""
if __name__ == "__main__":
    args = argparse_args()
    args.logger = utils.setup_logger(args.output_directory, args.verbose)
    args.logger.info(f"Output directory: {args.output_directory}")
    main(args)