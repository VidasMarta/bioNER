import requests
import argparse
import pandas as pd
import tqdm
import nltk
import os
import re
import numpy as np
import sys
import logging

from typing import Optional
from typing import List, Dict, Any, Optional
from datetime import datetime
#Use spacy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils


"""
This script sends a formatted chat prompt to a llama.cpp HTTP server for testing various biomedical and clinical NLP tasks.
It supports different system and prompt templates for information extraction, summarization, and diagnosis from medical text.
You can provide input via a CSV file of transcriptions or use a default example.
"""

def format_chat(messages: List[Dict[str, Any]]) -> str:
    """Format messages using the template compatible with llama.cpp server."""
    formatted = ""
    for i, msg in enumerate(messages):
        content = f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content'].strip()}<|eot_id|>"
        if i == 0:
            content = "<|begin_of_text|>" + content
        formatted += content
    # Add generation prompt for assistant
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted

def argparse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Llama.cpp HTTP server with a chat message.")
    parser.add_argument("--system_template", type=str, default="openbiollm-general",
        help="Choose from translation") 
    parser.add_argument("--csv", type=str, default=None, 
        help="Path to a file containing the text files for analysis and output in structured format.")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8080",
        help="URL of the llama.cpp HTTP server. If inside a container use 172.17.0.1 to acess outside server.")
    parser.add_argument("--verbose", action="store_true",
        help="Enable verbose output for debugging.")
    parser.add_argument("--max_tokens", type=int, default=100000,
        help="Maximum number of tokens to generate in the response.")
    parser.add_argument("--column_to_translate", type=str,
                        default='all',)
    parser.add_argument("--temperature", type=float, default=0.2,
        help="Temperature for the response generation, controlling randomness.")
    parser.add_argument("--output_dir", type=str, required=True,
        help="Directory to save the output files.")
    parser.add_argument("--reprocess", action="store_true",
        help="Enable reprocessing of translations.")
    parser.add_argument("--test", action="store_true",
        help="Enable testing on 10 samples.")
    return parser.parse_args()
"""python3 src/translateClinicalNotes.py --csv data/100_samples_medical_specialty_stratified_seed_42.csv\
    --system_template translation --temperature 0.1 --max_tokens 20000\
        --server_url http://172.17.0.1:8484
        
python3 src/translateClinicalNotes.py --csv /workspace/data/agbonet/annotation/100_samples_kmeans_30_seed_42.csv\
    --system_template translation --temperature 0 --max_tokens 20000\
        --server_url http://172.17.0.1:8484 --column_to_translate full_note
        
python3 src/translateClinicalNotes.py --csv /workspace/data/agbonet/agbonet.csv\
    --system_template translation --temperature 0 --max_tokens 5000\
        --server_url http://172.17.0.1:8484 --column_to_translate full_note\
            --output_dir /workspace/data/agbonet

python3 src/translateClinicalNotes.py --csv /workspace/data/agbonet/agbonet.csv\
    --system_template translation --temperature 0 --max_tokens 20000\
        --server_url http://0.0.0.0:8484 --column_to_translate full_note\
            --output_dir /workspace/data/agbonet

python3 src/translateClinicalNotes.py --csv data/mtsamples/mtsamples.csv\
    --system_template translation --temperature 0 --max_tokens 5000\
        --server_url http://localhost:8484 --column_to_translate transcription\
            --output_dir data/mtsamples --test
        
python3 src/translateClinicalNotes.py --csv data/100_samples_medical_specialty_stratified_seed_42.csv\
    --system_template translation --temperature 0.24 --max_tokens 200000\
        --server_url http://127.0.0.1:8484

python3 src/translateClinicalNotes.py \
    --csv /workspace/data/agbonet/annotation/100_samples_kmeans_30_seed_42.csv\
        --system_template translation --temperature 0.1 --max_tokens 20000\
            --server_url http://172.17.0.1:8484 --column_to_translate full_note\
                --output_dir /workspace/data/agbonet/annotation
"""

def sentence_split(text, min_chars=900, max_chars=1100):
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        if len(current_chunk) + len(sent) <= max_chars:
            current_chunk += (" " if current_chunk else "") + sent
        else:
            if len(current_chunk) >= min_chars:
                chunks.append(current_chunk.strip())
                current_chunk = sent
            else:
                # too short: try to squeeze in one more
                current_chunk += (" " if current_chunk else "") + sent
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def manual_postprocessing(df_hr: pd.DataFrame) -> pd.DataFrame:
    # Best would be to check sentences if they are hr or en 
    # and if there are more than 30% of en sentences drop that example
    mask = df_hr.map(lambda x: isinstance(x, str) and x != '' and (
        'patient' in str(x).lower() or 'date of birth' in str(x).lower() or
        'surgery' in str(x).lower() or 'medical history' in str(x).lower() or
        'removed' in str(x).lower() or 'discharged' in str(x).lower() or 
        'bunion' in str(x).lower() or 'translation' in str(x).lower()))
    for col in df_hr.columns:
        for idx in df_hr.index:
            if mask.loc[idx, col]:
                print(df_hr.loc[idx, col])
                df_hr.loc[idx, col] = np.nan
    return df_hr

def clean_text(text: str) -> str:
    text = re.sub(r'```+', '', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?,])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<\|.*?\|>', '', text)

    return text.strip()

def message_request(args: argparse.Namespace, chunk: str, col: str, i: int) -> requests.Response:
    system_templates = {
      'translation': """You are a medical translator. Translate the text to the 
        Croatian language while preserving medical terminology and context. 
        Ensure the translation is accurate and maintains the original meaning. 
        Take your time to ensure the target language is correct Croatian language 
        with 
        
        'dijalektalna karakteristika hrvatskog jezika u kojem se praslavenski 
        glas "jat" (ě) u dugim slogovima zamjenjuje sa "ije", a u kratkim slogovima 
        sa "je" koristeći hrvatsko štokavsko narječje' 
        
        parts and 'kirurgija' for 
        surgery, 'bubrežni kamenac' for kidney stone and 'DRENAŽA' for DRAIN. Don't 
        offer any extra informatioon just translate the text to Croatian language. 
        Use plain text format. Make shure you translate every single word to 
        croatian language even words written in CAPITAL letters.""",
      'check_translation': """You are checking translation from english to croatian. 
        You get only the croatian text and you need to ensure all the english words were translated in coratian. 
        Don't offer any extra information just check that all the words are in Croatian language. 
        Make shure you check every single word to be croatian language even words written in CAPITAL letters.
        Transcribe the text with all the words written in Croatian language. """,
      }
    messages = [
                    {"role": "system", 
                     "content": system_templates[args.system_template]},
                    {"role": "user", 
                     "content": 'Translate the text to croatian language:\n\n' +  chunk + '\n\n'}, ]
    prompt = format_chat(messages)
    # Send to llama.cpp HTTP server
    response = requests.post(
        f"{args.server_url}/completion",
        json={
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stop": ["<|eot_id|>"]
        })
    if args.verbose:
        args.logger.info('Translation response:')
        args.logger.info(f'For column {col} and row {i}, response status code: {response.status_code}')
        # print(f'For column {col} and row {i}, english text: {row[col]}')
        # Output response
        args.logger.info(response.json()['content'])
    if response.status_code != 200:
        args.logger.info(f"Error: {response.status_code} - {response.text}")
        raise Exception(f"Request failed with status code {response.status_code}")
    return response

def main(args: argparse.Namespace):
    args.logger.info(f"Arguments: {vars(args)}")
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.test: df = df.sample(n=10, random_state=42)
    else:
        # Example DataFrame for testing
        df = pd.DataFrame({
            args.column_to_translate : ["""The patient has diabetes and hypertension. 
                                        They are experiencing fatigue and frequent urination. 
                                        The doctor recommends a blood test to check glucose levels."""]})
    output_file = os.path.join(args.output_dir, f'{len(df)}_samples_hr_{args.temperature}_temp_{args.max_tokens}_max_tokens.csv')
    if args.column_to_translate == 'all':
        columns = df.columns
    else:
        if type(args.column_to_translate) == str:
            columns = [args.column_to_translate]
            if args.column_to_translate not in df.columns: 
                args.logger.info('ERROR wrongly specified column')
                exit(1)
            df.drop_duplicates(subset=columns, inplace=True)
        elif type(args.column_to_translate) == list:
            columns = args.column_to_translate
            for col in columns:
                if col not in df.columns: 
                    args.logger.info('ERROR wrongly specified column')
                    exit(1)
        else:
            args.logger.info('ERROR args wrongly defined. Wrongly specified column.')
            exit(1)
    if os.path.exists(output_file) and not args.reprocess:
        df_hr = pd.read_csv(output_file)
    else:
        df_hr = pd.DataFrame(columns=[col+ '_hr' for col in columns])

    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Translating"):
        if i < len(df_hr) and not args.reprocess:
            continue
        if not df_hr[df_hr['uid_hr'] == row['uid']].empty and not args.reprocess:
            continue
        df_tmp = pd.DataFrame(columns=[col + '_hr' for col in columns])
        for col in columns:
            if row[col] is None or pd.isna(row[col]) or type(row[col]) is not str or row[col].strip() == "":
                # print(f"Skipping column {col} in row {i} due to None or NaN value. It is {row[col]}")
                df_hr.at[i, col + '_hr'] = row[col]
                continue
            chunks = [row[col]]
            if len(row[col]) > 2000:
                # split the text into chunks of 1000 characters last chunk can be bigger than 1000
                # make shure to split on the last full sentence or just a . character
                chunks = sentence_split(row[col], 1000)
            final_text = ""
            for j, chunk in enumerate(chunks):
                response = message_request(args, chunk, col, i)
                text = response.json()['content'].strip()
                text = clean_text(text)
                lang = utils.check_language(text=text, method='all')
                if lang is not None and 'en' in lang:
                    args.logger.info(f"Detected language: {lang}, Text: {text}")
                    pass # Model halucinated in english
                else:
                    final_text += text + ' '
                    final_text = clean_text(final_text)
            df_tmp[col + '_hr'] = final_text.strip()
            if args.system_template == 'check_translation':
                pass #TODO: implement check translation
        df_tmp['uid_hr'] = row['uid']
        df_hr = pd.concat([df_hr, df_tmp], axis=0, ignore_index=True)
        
        if i % 10 == 0:
            args.logger.info(f"Processed {i} rows. Saving intermediate results.")
            df_hr.to_csv(output_file,index=False)
            
    df_hr.to_csv(output_file, index=False)
    # Replace cells containing the word 'patient' ond some english words with NaN and print their content
    df_hr = manual_postprocessing(df_hr)
    df_hr.to_csv(output_file, index=False)
    
def setup_logger(args):
    log_dir = args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(log_dir, f"{date_str}_llm_translation.log")
    logger = logging.getLogger("llm_translation")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(fh)
    logger.propagate = False
    return logger

if __name__ == "__main__":
    args = argparse_args()
    args.logger = setup_logger(args)
    main(args)
                