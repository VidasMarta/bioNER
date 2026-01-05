import json
from spacy.tokens import Span, Doc
from spacy.training.example import Example


def decode_tags(tag_ids):
    mapping = {0: "B", 1: "I", 2: "O"}
    return [mapping[t] for t in tag_ids]


def bio_to_spans(tokens, bio_tags, label="DISEASE"):
    spans = []
    start = None

    for i, tag in enumerate(bio_tags):
        if tag == "B":
            if start is not None:
                spans.append((start, i, label))
            start = i

        elif tag == "I":
            continue

        else:  # "O"
            if start is not None:
                spans.append((start, i, label))
                start = None

    if start is not None:
        spans.append((start, len(tokens), label))

    return spans


def load_data(nlp, jsonl_path):
    examples = []

    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)

            tokens = item["tokens"]
            tag_ids = item["tags"]

            # ---- FIX 1: remove empty tokens AND keep tags aligned ----
            clean_tokens = []
            clean_tags = []
            for tok, tag in zip(tokens, tag_ids):
                if isinstance(tok, str) and tok.strip():
                    clean_tokens.append(tok)
                    clean_tags.append(tag)

            if not clean_tokens:
                continue

            bio_tags = decode_tags(clean_tags)
            spans = bio_to_spans(clean_tokens, bio_tags, label="DISEASE")

            # ---- FIX 2: build token-based Doc ----
            doc = Doc(nlp.vocab, words=clean_tokens)

            ents = []
            for start, end, label in spans:
                if 0 <= start < end <= len(doc):
                    ents.append(Span(doc, start, end, label=label))

            doc.ents = ents

            # ---- FIX 3: gold doc is a COPY (no text alignment) ----
            gold = doc.copy()
            gold.ents = ents

            example = Example(doc, gold)
            examples.append(example)

    return examples
