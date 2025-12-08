import argparse
import os
import json
from typing import List, Optional, Tuple

from src_generate import promptGeneration, utils

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

def check_generated_size(args: argparse.Namespace, text: str) -> List[str]:
    nlp = spacy_load_model(args.spacy_model)
    doc = nlp(text)
    # if LLM generated more than one sentence return empty dict
    if len(list(doc.sents)) > args.num_sentences: 
        print('Multiple sentences loop of LLM generation error.')
        print(f'[INFO] Multiple sentences loop of LLM generation error: \n Sentences: {text}')
        # remove sentences after args.num_sentences
        doc = list(doc.sents)[:args.num_sentences]
        doc = nlp(" ".join([str(s) for s in doc]))
    if len(list(doc.sents)) > 1: 
        if list(doc.sents)[-2] == list(doc.sents)[-1]: 
            doc = list(doc.sents)[0]
            doc = nlp(str(doc))
    return [sent.text for sent in doc.sents if sent.text.strip()]


def create_rule_json(doc, nlp, term, entities)-> Tuple[list,list]:      
    tokens = [token.text for token in doc]
    tokens_lower = [token.lower() for token in tokens]
    tokens_lower = utils.check_last_token(tokens_lower)
    tags = [2] * len(tokens)  # default all "O" = 2
    # tokenize the term with spaCy as well (so alignment is consistent)
    term_tokens = [t.text.lower() for t in nlp(term.lower())]
    term_len = len(term_tokens)
    #TODO: is the lemmatization inside utils.check_last_token needed here?
    # search for the term sequence in tokens
    for i in range(len(tokens) - term_len + 1):
        if tokens_lower[i:i+term_len] == term_tokens:
            entities.append(term)
            tags[i] = 0  # B-DISEASE
            for j in range(1, term_len):
                tags[i+j] = 1  # I-DISEASE
            # break  # stop after first match
    return tags, tokens

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


def create_json(args: argparse.Namespace, text: str, 
                term: Tuple[str, str]) -> List[dict]:
    # TODO: add proposed entities from parsing step Where and why?
    nlp = spacy_load_model(args.spacy_model)
    doc = nlp(text)
    entities = []
    tags, tokens = create_rule_json(doc, nlp, term, entities)
    terms = term #[term[0].lower()]
    #term_ids = [term[1]]
    if 0 in tags:
        merged_tags = tags  # default all "O" = 2
        terms_llm = check_additional_disease_tags(args, tokens, term[0].lower())
        # TODO provjeriti da nema više istih entiteta a nema ih u tekstu
        for term_llm in terms_llm:
            if term_llm[0].lower() != term[0].lower():
                terms.append(term_llm[0].lower())
                #term_ids.append('NaN')
                print(f'[INFO] Additional disease term found in generated text: {term_llm} for original term {term}.')
                print(f'[INFO] Generated sentence: {text}')
                tags_llm, _ = create_rule_json(doc, nlp, term_llm, entities)
                for i in range(len(tags_llm)):
                    if tags_llm[i] == 0:  # B-DISEASE
                        merged_tags[i] = 0
                    elif tags_llm[i] == 1 and merged_tags[i] != 0:  # I-DISEASE
                        merged_tags[i] = 1
        tags = merged_tags

    pos_tags = [token.pos_ for token in doc]
    dep_rels = [token.dep_ for token in doc]
    parents = [token.head.i for token in doc] #index of parent token
    corpus = "generated_train"
    json_data = {
        "abstract_id": None,
        "sentence": text,
        "entities": entities,
        "corpus": corpus,
        "pos": pos_tags, 
        "dep": dep_rels, 
        "parents": parents,
        "tags": tags, 
        "tokens": tokens, 
    }

    return json_data

def load_data(args: argparse.Namespace):
    if os.path.exists(args.generated):
        with open(args.generated, "r") as f:
            new_data = [json.loads(line) for line in f]

    print("[INFO] Loaded generated data for post processing.")
    with open(args.generated_postprocessed, "w") as f:
        id = args.starting_id
        for data in new_data:
            text = data["text"]
            term = data["term"]
            sentences = check_generated_size(args, text)
            for sent in sentences:
                text_json = create_json(args, sent, term)
                text_json["id"] = id
                f.write(json.dumps(text_json))
                f.write("\n")
                id += 1
    

def argparse_args():
    parser = argparse.ArgumentParser(description="LLM-based text Generator iteration pipeline with k-shot.")
    parser.add_argument('--generated', type=str, default='',
                        help='Path to the generated sentences.')
    parser.add_argument('--generated_postprocessed', type=str, default='',
                        help='Path to the generated sentences after postprocessing.')
    parser.add_argument('--starting_id', type=int, help='Id of last generated sentence.')
    parser.add_argument('--num_sentences', type=int, help='Number of sentences LLM was supposed to generate.')
    parser.add_argument('--spacy_model', type=str, default="en_core_web_sm")

    return parser.parse_args()

if __name__ == "__main__":
    args = argparse_args()
    SPACY_NLP = spacy_load_model(args.spacy_model)
    load_data(args)