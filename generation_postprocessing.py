import argparse
import os
import json
from typing import List, Optional, Tuple, Any

import yaml

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
        args.logger.info(f'Multiple sentences loop of LLM generation error: \n Sentences: {text}')
        # remove sentences after args.num_sentences
        doc = list(doc.sents)[:args.num_sentences]
        doc = nlp(" ".join([str(s) for s in doc]))
    if len(list(doc.sents)) > 1: 
        if list(doc.sents)[-2] == list(doc.sents)[-1]: 
            doc = list(doc.sents)[0]
            doc = nlp(str(doc))
    return [sent.text for sent in doc.sents if sent.text.strip()]


def create_rule_json(doc, nlp, term_tuples: List[Tuple[List, List]], 
                     tags: list = [])-> Tuple[list, list, list, list]: 
    tokens = [t.text for t in doc]
    tokens_clean = utils.check_last_token(tokens)  
    tokens_lower = [t.lower() for t in tokens_clean]
    entities = []   
    
    if tags: pass
    else: tags = [2] * len(tokens)  # default all "O" = 2
    term_tokens_string = []
    for term, idx in term_tuples:
        term = term[0].lower()       # FIX 1
        # tokenize the term with spaCy as well (so alignment is consistent)
        term_tokens = [t.text.lower() for t in nlp(term)]
        term_tokens_string.append(' '.join(term_tokens))
        term_len = len(term_tokens)
        #TODO: is the lemmatization inside utils.check_last_token needed here?
        for i in range(len(tokens) - term_len + 1):
            if tokens_lower[i:i+term_len] == term_tokens:
                
                entities.append(term)
                tags[i] = 0  # B-DISEASE
                for j in range(1, term_len):
                    tags[i+j] = 1  # I-DISEASE

    return tags, tokens, entities, term_tokens_string

def check_additional_disease_tags(args: argparse.Namespace, text: str, term, 
                                  system_template: str = 'annotation', 
                                  user_template: str = 'disease_annotation',) -> str:
    # text = " ".join(tokens)
    args.logger.info("ANNOTATION FOR:")
    args.logger.info(f" Generated SENTENCE: {text}")
    
    response = promptGeneration.message_request(args, term,
                system_template=system_template, 
                user_template=user_template,
                text=text)
    # print('+'*80)
    # print(text)
    # pprint.pprint(eval(response.json()['content']))
    cleaned_content = utils.remove_code_fences(response.json()['content']) 
    args.logger.info(f"Predicted Entities: {cleaned_content}")
       
    try:
        term_llm = eval(cleaned_content)
        return [term for term in term_llm] #return [(term, 'NaN') for term in term_llm]
    except Exception as e:
        return term #[(term, 'NaN')]


def append_token_wise_different_entities(ents, entities_all, nlp):
    entities_all_strings = []
    ents_strings = []
    # Tokenize terms and then check if the joined string of tokenization is the same
    for term in ents:
        ents_strings.append(' '.join([t.text.lower() for t in nlp(term)]))    
    for term in entities_all:
        entities_all_strings.append(' '.join([t.text.lower() for t in nlp(term)]))
        
    for ent, true_ent in zip(ents_strings, ents):
        if ent in entities_all_strings:
            pass
        else: 
            entities_all.append(true_ent)
    return entities_all


def create_json(args: argparse.Namespace, 
                text: str, 
                terms: Tuple[List[str], List[str]], 
                nlp: Any = None,
                system_template: str = 'annotation',
                user_template: str = 'disease_annotation_reduced'):
    # TODO: add proposed entities from parsing step Where and why?
    if nlp:
        nlp = nlp
    else: 
        nlp = spacy_load_model(args.spacy_model)
    doc = nlp(text)
    #TODO: Unpack the terms if they are len >1
    unpacked_terms = []
    print('Usual terms are: ', terms )
    if len(terms[0]) > 1:
        print(terms)
        for t in zip(terms[0],terms[1]):
            unpacked_terms.append(([t[0]], [t[1]])) 
        terms = unpacked_terms 
        print('\n Unpacked terms are:', unpacked_terms)
    else: 
        unpacked_terms.append(terms)
        terms = [terms]
    
    tags, tokens, entities, entities_strings = create_rule_json(doc, nlp, terms)
    entities_all = entities
    
    term_ids = [uid[0] for _, uid in terms]
    # Unpack the terms to generate an check for additional disease tags.
    terms = [t[0].lower() for t, _ in terms]
    
    # if 0 in tags: #TODO: Why is this condition of 0 here?????
    # merged_tags = tags  # default all "O" = 2
    args.logger.info(f'Terms before LLM check: {terms}')
    
    terms_llm = check_additional_disease_tags(args, text, terms, 
                                              system_template=system_template, 
                                              user_template=user_template)
    terms_llm = [([t], []) for t in terms_llm if t[0].lower() not in terms]
    tags_llm, _, ents, ents_strings = create_rule_json(doc, nlp, terms_llm)
    # TODO provjeriti da nema više istih entiteta a nema ih u tekstu
    # entities_all = append_token_wise_different_entities(ents, entities_all, nlp)
    
    # Another check to add entitie to entities all or not
    for ent, true_ent in zip(ents_strings, ents):
        if ent in entities_strings:
            pass
        else: 
            unpacked_terms.append(([true_ent], []))
            entities_all.append(true_ent)
    args.logger.info(f'Final entities list: {entities_all}')

    # Merge tags (BIO) (012) for new entities
    for i in range(len(tags_llm)):
        if tags_llm[i] == 0:  # B-DISEASE
            tags[i] = 0
        elif tags_llm[i] == 1 and tags[i] != 0:  # I-DISEASE
            tags[i] = 1

    print('\n\n Unpacked terms are:', unpacked_terms)
    
    pos_tags = [token.pos_ for token in doc]
    dep_rels = [token.dep_ for token in doc]
    parents = [token.head.i for token in doc] #index of parent token
    # print('Entities_ALL ARE: ', entities_all)
    json_data = {
        "abstract_id": None,
        "sentence": text,
        "entities": entities_all,
        "corpus": "generated_train",
        "pos": pos_tags, 
        "dep": dep_rels, 
        "parents": parents,
        "tags": tags, 
        "tokens": tokens, 
        "terms": unpacked_terms,
    }

    return json_data

def load_data(args: argparse.Namespace):
    if os.path.exists(args.generated):
        with open(args.generated, "r") as f:
            new_data = [json.loads(line) for line in f]

    #print("[INFO] Loaded generated data for post processing.")
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
    parser.add_argument('--config_file', type=str, help='Number of sentences LLM was supposed to generate.', default='/home/mvidas/syn-bioner/bioNER/experiments/default_generate.yml')

    return parser.parse_args()

if __name__ == "__main__":
    init_args = argparse_args()
    with open(init_args.config_file, 'r') as file:
        yaml_args = yaml.safe_load(file)
    args = argparse.Namespace(**yaml_args)
    args.generated = init_args.generated
    args.starting_id = init_args.starting_id
    args.generated_postprocessed = init_args.generated_postprocessed
    logger = utils.setup_logger(args.output_directory, args.verbose)
    args.logger = logger
    SPACY_NLP = spacy_load_model(args.spacy_model)
    load_data(args)