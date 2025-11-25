import json
import pprint
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tqdm
import collections
import os
import re
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict

import networkx as nx  

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
    parser.add_argument('--system_prompt_key', type=str, default='generation', 
                        help="Key for selecting system prompt from predefined prompt templates (default: 'generation').")
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
    parser.add_argument('--spacy_model', type=str, default='en_core_web_sm', 
                        help='spaCy model to use for tokenization (default: en_core_web_sm).')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for reproducibility.')
    parser.add_argument('--include_pos', action='store_true', 
                        help='Include POS tags in k-shot examples.')
    parser.add_argument('--include_dep', action='store_true', 
                        help='Include dependency tags in k-shot examples.')
    parser.add_argument('--no_entity_ratio', type=float, default=0.25,
                        help='Ratio of sentences without entities.')


    return parser.parse_args()

SPACY_NLP = None

def spacy_load_model(model_name: str):

    global SPACY_NLP
    if SPACY_NLP is None:
        SPACY_NLP = get_spacy(model_name)
        return SPACY_NLP
    return SPACY_NLP   

def get_spacy(model_name: str):
    import spacy
    try: return spacy.load(model_name)
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        os.system(f"python3 -m spacy download {model_name}")
        print(f"Downloading {model_name} model")
        import spacy
        return spacy.load(model_name)
    
def spacy_preprocess(text: str):
    nlp = spacy_load_model()
    doc = nlp(text)
    return doc, nlp


def create_rule_json(doc, nlp, term)-> Tuple[list,list]:         
    tokens = [token.text for token in doc]
    tokens_lower = [token.lower() for token in tokens]
    tokens_lower = utils.check_last_token(tokens_lower)
    tags = [2] * len(tokens)  # default all "O" = 2
    # tokenize the term with spaCy as well (so alignment is consistent)
    term_tokens = [t.text.lower() for t in nlp(term[0].lower())]
    term_len = len(term_tokens)
    #TODO: is the lemmatization inside utils.check_last_token needed here?
    # search for the term sequence in tokens
    for i in range(len(tokens) - term_len + 1):
        if tokens_lower[i:i+term_len] == term_tokens:
            tags[i] = 0  # B-DISEASE
            for j in range(1, term_len):
                tags[i+j] = 1  # I-DISEASE
            # break  # stop after first match
    return tags, tokens

def check_generated_size(args: argparse.Namespace, text: str) -> List[str]:
    nlp = spacy_load_model(args.spacy_model)
    doc = nlp(text)
    # if LLM generated more than one sentence return empty dict
    if len(list(doc.sents)) > args.num_sentences: 
        print('Multiple sentences loop of LLM generation error.')
        args.logger.info(f'Multiple sentences loop of LLM generation error: \n Sentences: {text}')
        # remove sentences after args.num_sentences
        doc = list(doc.sents)[:args.num_sentences]
        doc = nlp(" ".join([str(s) for s in doc]))
    if len(list(doc.sents)) > 1: 
        if list(doc.sents)[-2] == list(doc.sents)[-1]: 
            doc = list(doc.sents)[0]
            doc = nlp(str(doc))
    return [sent.text for sent in doc.sents if sent.text.strip()]

def create_json(args: argparse.Namespace, text: str, 
                term: Tuple[str, str], 
                entities: Optional[List[str]] = None) -> List[dict]:
    # TODO: add proposed entities from parsing step Where and why?
    nlp = spacy_load_model(args.spacy_model)
    doc = nlp(text)
    tags, tokens = create_rule_json(doc, nlp, term)
    terms = [term[0].lower()]
    term_ids = [term[1]]
    if 0 in tags:
        merged_tags = tags  # default all "O" = 2
        terms_llm = check_additional_disease_tags(args, tokens, term[0].lower())
        # TODO provjeriti da nema više istih entiteta a nema ih u tekstu
        for term_llm in terms_llm:
            if term_llm[0].lower() != term[0].lower():
                terms.append(term_llm[0].lower())
                term_ids.append('NaN')
                args.logger.info(f'Additional disease term found in generated text: {term_llm} for original term {term}.')
                args.logger.info(f'Generated sentence: {text}')
                tags_llm, tokens_llm = create_rule_json(doc, nlp, term_llm)
                for i in range(len(tags_llm)):
                    if tags_llm[i] == 0:  # B-DISEASE
                        merged_tags[i] = 0
                    elif tags_llm[i] == 1 and merged_tags[i] != 0:  # I-DISEASE
                        merged_tags[i] = 1
        tags = merged_tags

    json_data = {"tags": tags, "tokens": tokens, "term": terms, "term_id": term_ids}

    if 0 not in tags:
        args.logger.info(f'Term not found in generated text: {term}.')
        args.logger.info(f'Generated sentence: {text}')
        use_llm_annotation = True
        terms_llm = check_additional_disease_tags(args, tokens, term[0].lower())
        if len(terms_llm) > 1:
            multiple_tags = []
            multiple_tokens = []
            for term_llm in terms_llm:
                tags, tokens = create_rule_json(doc, nlp, term_llm)
                multiple_tags.append(tags)
                multiple_tokens.append(tokens)
            merged_tags = [2] * len(tokens)  # default all "O" = 2
            for tags in multiple_tags:
                for i in range(len(tags)):
                    if tags[i] == 0:  # B-DISEASE
                        merged_tags[i] = 0
                    elif tags[i] == 1 and merged_tags[i] != 0:  # I-DISEASE
                        merged_tags[i] = 1
            tags = merged_tags
        elif len(terms_llm) == 1 and terms_llm[0][0].lower() != term[0].lower():
            tags, tokens = create_rule_json(doc, nlp, terms_llm[0])       
        
        json_data = {"tags": tags, "tokens": tokens, "term": [term[0] for term in terms_llm], "term_id": [term[1] for term in terms_llm]}
        if 0 not in tags:
            args.logger.info(f'No disease term found in generated text even after LLM check: {terms_llm}.')
            args.logger.info(f'Generated sentence: {text}')
            # return {}
    if len(tags) != len(tokens):
        print('JSON creation error.')
        print(text)
        args.logger.info(f'Term JSON creation error wrong lengths of sequences: {term}.')
        args.logger.info(f'Generated sentence: {text}')
        
        return {}
            
    if args.verbose:
        print('+'*80)
        print(f'For text: {text} in create json function:')
        print('Lenght of tags is: ', len(tags))
        print('Lenght of tokens is: ', len(tokens))
        print(f'The tags should be: {tags}')
        print(f'The tokens should be: {tokens}')
        print(f'The term should be: {term}')
    
    return json_data

def check_additional_disease_tags(args: argparse.Namespace, tokens, term) -> str:
    text = " ".join(tokens)
    response = promptGeneration.message_request(args, term,
                system_template='annotation', 
                user_template='disease_annotation',
                text=text)
    # print('+'*80)
    # print(text)
    # pprint.pprint(eval(response.json()['content']))
    cleaned_content = utils.remove_code_fences(response.json()['content'])    
    try:
        term_llm = eval(cleaned_content)
        return [(term, 'NaN') for term in term_llm]
    except Exception as e:
        return [(term, 'NaN')]
    
def generate_sentence_samples(
    args: argparse.Namespace,
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

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    date_today = datetime.today().strftime("%Y%m%d")
    np.random.seed(getattr(args, "random_seed", 42))  # Fixed seed for reproducibility

    output_path = os.path.join(args.output_directory, f'generated_sentences_{date_today}.json')

    # -------------------------------------------------------------------------
    # Load k-shot examples
    # -------------------------------------------------------------------------
    kshot_examples = []
    if args.kshot_path and os.path.exists(args.kshot_path):
        kshot_examples = utils.load_training_samples(args.kshot_path)
        if args.verbose:
            print(f"[INFO] Loaded {len(kshot_examples)} k-shot examples from {args.kshot_path}")
    else:
        print(f"[INFO] No k-shot examples loaded. BUG! Check path: {args.kshot_path}")
        print(f"{os.path.exists(args.kshot_path)} {os.getcwd()}")
    # TODO: sampling logic for k-shot examples
    # Filter by havig entity or not in a kshot_examples pool
    entity_examples = [ex for ex in kshot_examples if ex.get("entities")]
    no_entity_examples = [ex for ex in kshot_examples if not ex.get("entities")]

    # -------------------------------------------------------------------------
    # Generation loop
    # -------------------------------------------------------------------------
    for i, term in enumerate(tqdm.tqdm(term_list)):
        with open(output_path, method, encoding='utf-8') as file:
            try:
                # -----------------------------------------------------------------
                # Sample k examples randomly for this term
                # -----------------------------------------------------------------
                if kshot_examples:
                    if np.random.rand() < args.no_entity_ratio:
                        # generate sentence with no entity
                        kshot_text_block, used_ids = utils.sample_k_examples(args, no_entity_examples)
                        user_template = 'kshot_genre_no_entity'
                    else:
                        # generate sentence with an entity
                        kshot_text_block, used_ids = utils.sample_k_examples(args, entity_examples)
                        user_template = 'kshot_num_sent_genre_entity'
                        # shot_text = promptGeneration.PROMPT[user_template].format(
                        #     number_of_sentences=args.num_sentences,
                        #     condition=term[0],
                        #     genre=promptGeneration.Genre.ABSTRACT.value,
                        #     text=kshot_text_block
                        # )

                    args.logger.info(f"K-shot examples used for term '{term[0]}': {used_ids}")
                else:
                    # Fallback single-shot mode
                    user_template = 'syn_generation'
                    used_ids = []

                # -----------------------------------------------------------------
                # Send request to LLM
                # -----------------------------------------------------------------
                response = promptGeneration.message_request(
                        args,
                        term[0],
                        system_template=system_template,
                        user_template=user_template,
                        text=kshot_text_block
                    )
                # print(response.json())
                text = response.json()['content'].strip()
                text = utils.clean_text(text)
                text = utils.remove_code_fences(text)
                # TODO: parsing Sentence: ... Entities: [...] format
                text, entities = utils.parse_text_entities_format(args, text)
                if args.verbose:
                    args.logger.info(f"Generated text is: {text}")
                    args.logger.info(
                        f"Extracted entities proposed by LLM: {entities}\n Term is: {term[0]}")
                sentences = check_generated_size(args, text)
                # -----------------------------------------------------------------
                # Write generated sentences as JSON
                # -----------------------------------------------------------------
                for sent in sentences:
                    # TODO: add proposed entities from parsing step
                    text_json = create_json(args, sent, term, entities=entities)
                    if text_json:
                        # Store metadata about k-shot context
                        text_json["kshot_example_ids"] = used_ids
                        text_json["include_pos"] = getattr(args, "include_pos", True)
                        text_json["include_dep"] = getattr(args, "include_dep", True)
                        text_json["random_seed"] = getattr(args, "random_seed", 42)

                        file.write(json.dumps(text_json))
                        file.write("\n")

            except Exception as e:
                args.logger.info(f"Failed to generate or parse sentence for term {term}: {e}")
                args.logger.info(f"Response content: {response.json().get('content', '')}")
                if args.verbose:
                    print(f"[ERROR] Term {term}: {e}")
    return output_path

    

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

    generate_sentence_samples(args, term_list)
    
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
    SPACY_NLP = spacy_load_model(args.spacy_model)
    args.logger = setup_logger(args)
    args.logger.info(f"Output directory: {args.output_directory}")
    main(args)