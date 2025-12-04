import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import random
import json
from pathlib import Path
from spacy.tokens import Span, Doc
from spacy_ner_utils import *

def train(output_dir, train_path, n_iter=20):
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")


    train_examples = load_data(nlp, train_path)

    # Add labels
    for example in train_examples:
        for ent in example.reference.ents:
            ner.add_label(ent.label_)

    nlp.begin_training()

    for it in range(n_iter):
        random.shuffle(train_examples)
        losses = {}

        batches = minibatch(train_examples, size=8)
        for batch in batches:
            nlp.update(batch, drop=0.3, losses=losses)

        print(f"Iter {it}: Loss = {losses}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"[INFO] Saved model to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_iter", type=int, default=20)
    args = parser.parse_args()

    train(args.output, args.train, args.n_iter)
