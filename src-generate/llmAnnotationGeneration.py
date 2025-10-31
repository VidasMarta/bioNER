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
import spacy
import logging
from datetime import datetime
from collections import defaultdict

import networkx as nx  

import promptGeneration
import utils


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
    parser.add_argument('--spacy_model', type=str, default='en_core_web_sm', 
                        help='spaCy model to use for tokenization (default: en_core_web_sm).')
    parser.add_argument('--training_examples_file', type=str, default='', 
                        help='Path to the file containing training examples for few-shot prompting.')
    parser.add_argument('--kshot_path', type=str, default='', 
                        help='Path to the file containing training examples for few-shot prompting.')
    parser.add_argument('--kshot_size', type=int, default=0, 
                        help='Number of examples to use for each prompt.')
    parser.add_argument('--spacy_model', type=str, default='en_core_web_sm', 
                        help='spaCy model to use for tokenization (default: en_core_web_sm).')

    return parser.parse_args()

SPACY_NLP = None

def spacy_load_model(model_name: str):

    global SPACY_NLP
    if SPACY_NLP is None:
        SPACY_NLP = spacy.load(model_name)
        return SPACY_NLP
    return SPACY_NLP   

def spacy_preprocess(text: str):
    nlp = spacy_load_model()
    doc = nlp(text)
    return doc, nlp

# def format_chat(messages: List[Dict[str, Any]]) -> str:
#     """Format messages using the template compatible with llama.cpp server."""
#     formatted = ""
#     for i, msg in enumerate(messages):
#         content = f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content'].strip()}<|eot_id|>"
#         if i == 0:
#             content = "<|begin_of_text|>" + content
#         formatted += content
#     # Add generation prompt for assistant
#     formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
#     return formatted

# def message_request(args: argparse.Namespace, chunk: str, 
#                     i: int, prompt_key: str) -> requests.Response:
#     messages = [ {"role": "system", "content": SYSTEM_PROMPTS[args.system_prompt_key]},
#                     {"role": "user", "content": PROMPT[prompt_key] +  f"""
#                      Consider the sentence output that comes after sentence: 
     
#         The patient diagnosed with {chunk}. 
#         Based on your medical expertise. Your task is to generate next sentence containing following disease: """ +  chunk + '\n\n'}, ]
    
#     prompt = format_chat(messages)
#     # Send to llama.cpp HTTP server
#     response = requests.post(
#         f"{args.server_url}/completion",
#         json={
#             "prompt": prompt,
#             "max_tokens": args.max_tokens,
#             "temperature": args.temperature,
#             "stop": ["<|eot_id|>"]
#         })
#     if args.verbose:
#         args.logger.info('Translation response:')
#         args.logger.info(f'For column and row {i}, response status code: {response.status_code}')
#         # Output response
#         args.logger.info(response.json()['content'])
#     if response.status_code != 200:
#         args.logger.info(f"Error: {response.status_code} - {response.text}")
#         raise Exception(f"Request failed with status code {response.status_code}")
#     return response

def create_rule_json(doc, nlp, term)-> Tuple[list,list]:         
    tokens = [token.text for token in doc]
    tokens_lower = [token.lower() for token in tokens]
    tokens_lower = utils.check_last_token(tokens_lower)
    tags = [2] * len(tokens)  # default all "O" = 2
    # tokenize the term with spaCy as well (so alignment is consistent)
    term_tokens = [t.text.lower() for t in nlp(term[0].lower())]
    term_len = len(term_tokens)

    # search for the term sequence in tokens
    for i in range(len(tokens) - term_len + 1):
        if tokens_lower[i:i+term_len] == term_tokens:
            tags[i] = 0  # B-DISEASE
            for j in range(1, term_len):
                tags[i+j] = 1  # I-DISEASE
            # break  # stop after first match
    return tags, tokens

def check_generated_size(args: argparse.Namespace, text: str) -> List[str]:
    nlp = spacy_load_model()
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

def create_json(args: argparse.Namespace, text: str, term: Tuple[str, str]) -> List[dict]:
    nlp = spacy_load_model(args.spacy_model)
    doc = nlp(text)
    tags, tokens = create_rule_json(doc, nlp, term)
    terms = [term[0].lower()]
    term_ids = [term[1]]
    if 0 in tags:
        merged_tags = tags  # default all "O" = 2
        terms_llm = check_additional_disease_tags(args, tokens, term[0].lower())
        # TODO provjeriti da nema više sitih entiteta a nema ih u tekstu
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
        
        # TODO add the non entity sentence   
        json_data = {"tags": tags, "tokens": tokens, "term": [term[0] for term in terms_llm], "term_id": [term[1] for term in terms_llm]}
        if 0 not in tags:
            args.logger.info(f'No disease term found in generated text even after LLM check: {terms_llm}.')
            args.logger.info(f'Generated sentence: {text}')
            return {}
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
    try:
        term_llm = eval(response.json()['content'])
        return [(term, 'NaN') for term in term_llm]
    except Exception as e:
        return [(term, 'NaN')]

def generate_sentence_samples(args: argparse.Namespace, term_list: List[str], method: str='a',
                              system_template: str='role_prompt', 
                              user_template: str='genre_prompt'):
    """
    Generate sentences for a given list of terms. And for a given samples if args.kshot_path is provided.
    Each generated sentence is saved as a JSON object in the specified output directory.
    """

    date_today = datetime.today().strftime("%Y%m%d")
    if args.kshot_path: #TODO implement k-shot generation
        df = utils.load_training_samples(args.kshot_path)
        shots = utils.make_kshot(df, args.kshot_size)
        user_template = 'kshot_genre_generation'
        
    for i, term in enumerate(tqdm.tqdm(term_list)):
        with open(os.path.join(args.output_directory, 
                'generated_sentences_' + date_today + '.txt'), method) as file:
            shot_id = 
            response = promptGeneration.message_request(args, 
                                                        term[0], 
                                                        system_template=system_template,
                                                        user_template=user_template,
                                                        text=shot)
            text = response.json()['content'].strip()
            text = utils.clean_text(text)
            text = utils.remove_code_fences(text)
            sentences = check_generated_size(args, text)
            for text in sentences:
                # print(term)
                text_json = create_json(args, text, term)
                # check_additional_disease_tags(args, text_json)
                # Firstly for this json check for other disease mentions
                
                if text_json:
                    file.write(json.dumps(text_json))  
                    file.write('\n')   
        # except Exception as e:
        #     args.logger.info(f"Failed to parse response for sentence {i}: \n Exception: {e}")
    

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

""" python3 /home/mkeber/syn-bioner/src/llmAnnotationGeneration.py \
    --input_file /home/mkeber/syn-bioner/data/NCBI-Disease/val_wrong_ent.txt \
    --temperature 0 --max_tokens 500 --input_file_type list\
    --server_url http://0.0.0.0:8484 \
    --obo_file_path /home/mkeber/syn-bioner/HumanDiseaseOntology/src/ontology/HumanDO.obo \
    --output_directory /home/mkeber/syn-bioner/data/synthetic3 \
    --test --num_sentences 3

python3 /home/mkeber/syn-bioner/src/llmAnnotationGeneration.py \
    --input_file /home/mkeber/syn-bioner/data/SNOMEDCT/concepts_filtered.csv \
    --temperature 0 --max_tokens 500 --input_file_type list\
    --server_url http://0.0.0.0:8484 \
    --output_directory /home/mkeber/syn-bioner/data/synthetic-snomed \
    --test --num_sentences 3
    
python3 /home/mkeber/syn-bioner/src/llmAnnotationGeneration.py \
    --input_file /home/mkeber/syn-bioner/data/NCBI-Disease/val_wrong_ent.txt \
    --temperature 0 --max_tokens 500 --input_file_type list\
    --server_url http://172.17.0.1:8484 \
    --output_directory /home/mkeber/syn-bioner/data/synthetic2 \
    --num_sentences 3 --spacy_model en_core_web_lg \
    --test
"""
if __name__ == "__main__":
    args = argparse_args()
    SPACY_NLP = spacy.load(args.spacy_model)
    args.logger = setup_logger(args)
    args.logger.info(f"Output directory: {args.output_directory}")
    main(args)