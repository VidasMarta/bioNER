
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
            bio_tags = decode_tags(tag_ids)
            spans = bio_to_spans(tokens, bio_tags, label="DISEASE")

            doc = Doc(nlp.vocab, words=tokens)
            ents = [Span(doc, start, end, label=label) for start, end, label in spans]
            doc.ents = ents

            example = Example(doc, doc)  # gold-standard is same doc for now
            examples.append(example)
    return examples