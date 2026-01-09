import argparse
import os
import json
from typing import List, Tuple, Any

import yaml
from src_generate import prompt_generation, utils

SPACY_NLP = None

def spacy_load_model(model_name: str):
    global SPACY_NLP
    if SPACY_NLP is None:
        SPACY_NLP = get_spacy(model_name)
    return SPACY_NLP


def get_spacy(model_name: str):
    import spacy
    try:
        return spacy.load(model_name)
    except Exception:
        os.system(f"python3 -m spacy download {model_name}")
        return spacy.load(model_name)

def check_generated_size(args, text: str) -> List[str]:
    nlp = spacy_load_model(args.spacy_model)
    doc = nlp(text)

    sents = list(doc.sents)
    if len(sents) > args.num_sentences:
        sents = sents[:args.num_sentences]

    cleaned = []
    for s in sents:
        s = s.text.strip()
        if s:
            cleaned.append(s)

    return cleaned


def create_rule_json(
    doc,
    nlp,
    term_tuples: List[Tuple[List[str], List[str]]],
):
    tokens = [t.text for t in doc]
    tokens = utils.check_last_token(tokens)
    tokens_lower = [t.lower() for t in tokens]

    n_tokens = len(tokens)
    tags = [2] * n_tokens  # O

    # ---- collect candidate spans ----
    candidate_spans = []  # (start, end, text)

    for term, _ in term_tuples:
        if not term:
            continue

        term_text = term[0].strip().lower()
        if not term_text:
            continue

        term_tokens = [t.text.lower() for t in nlp(term_text)]
        L = len(term_tokens)
        if L == 0 or L > n_tokens:
            continue

        for i in range(n_tokens - L + 1):
            if tokens_lower[i:i+L] == term_tokens:
                candidate_spans.append((i, i + L, term_text))

    # ---- longest span wins ----
    candidate_spans.sort(key=lambda x: (x[1] - x[0]), reverse=True)

    final_spans = []
    occupied = set()

    for start, end, text in candidate_spans:
        span_range = set(range(start, end))
        if occupied.intersection(span_range):
            continue
        final_spans.append((start, end, text))
        occupied.update(span_range)

    # ---- BIO tagging ----
    for start, end, _ in final_spans:
        tags[start] = 0
        for i in range(start + 1, end):
            tags[i] = 1

    entities = [text for _, _, text in final_spans]

    return tags, tokens, entities, final_spans


def check_additional_disease_tags(
    args,
    text: str,
    terms: List[str],
    system_template: str = "annotation",
    user_template: str = "disease_annotation",
):
    response = prompt_generation.message_request(
        args,
        terms,
        system_template=system_template,
        user_template=user_template,
        text=text,
    )

    cleaned = utils.remove_code_fences(response.json()["content"])

    try:
        parsed = eval(cleaned)
        return [t.lower() for t in parsed if isinstance(t, str)]
    except Exception:
        return []


def create_json(
    args,
    text: str,
    id: str,
    terms: Tuple[List[str], List[str]],
    nlp: Any = None,
    system_template: str = "annotation",
    user_template: str = "disease_annotation_reduced",
):
    if nlp is None:
        nlp = spacy_load_model(args.spacy_model)

    doc = nlp(text)

    # ---- normalize terms ----
    unpacked_terms = []
    for t, tid in zip(terms[0], terms[1]):
        unpacked_terms.append(([t], [tid]))

    # ---- rule-based pass ----
    tags, tokens, entities, spans = create_rule_json(doc, nlp, unpacked_terms)

    # ---- LLM pass ----
    base_terms = [t[0].lower() for t, _ in unpacked_terms]
    llm_terms = check_additional_disease_tags(
        args, text, base_terms, system_template, user_template
    )

    llm_term_tuples = [([t], []) for t in llm_terms if t not in base_terms]

    tags_llm, _, entities_llm, spans_llm = create_rule_json(
        doc, nlp, llm_term_tuples
    )

    # ---- merge spans safely ----
    occupied = set(i for s, e, _ in spans for i in range(s, e))

    for start, end, text_llm in spans_llm:
        span_range = set(range(start, end))
        if occupied.intersection(span_range):
            continue
        spans.append((start, end, text_llm))
        entities.append(text_llm)
        occupied.update(span_range)

        tags[start] = 0
        for i in range(start + 1, end):
            tags[i] = 1

    pos_tags = [t.pos_ for t in doc]
    dep_rels = [t.dep_ for t in doc]
    parents = [t.head.i for t in doc]

    return {
        "abstract_id": None,
        "id": id,
        "sentence": text,
        "entities": entities,
        "corpus": "generated_train",
        "pos": pos_tags,
        "dep": dep_rels,
        "parents": parents,
        "tags": tags,
        "tokens": tokens,
        "terms": unpacked_terms,
    }

def load_data(args):
    with open(args.generated) as f:
        new_data = [json.loads(line) for line in f]

    with open(args.generated_postprocessed, "w") as f:
        idx = args.starting_id
        for item in new_data:
            sentences = check_generated_size(args, item["text"])
            for sent in sentences:
                js = create_json(args, sent, idx, item["term"])
                f.write(json.dumps(js) + "\n")
                idx += 1


def argparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str)
    parser.add_argument("--generated_postprocessed", type=str)
    parser.add_argument("--starting_id", type=int)
    parser.add_argument("--config_file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    init_args = argparse_args()
    with open(init_args.config_file) as f:
        cfg = yaml.safe_load(f)

    args = argparse.Namespace(**cfg)
    args.generated = init_args.generated
    args.generated_postprocessed = init_args.generated_postprocessed
    args.starting_id = init_args.starting_id

    args.logger = utils.setup_logger(args.output_directory, args.verbose)
    spacy_load_model(args.spacy_model)

    load_data(args)