import spacy
from spacy.scorer import Scorer
from spacy.training.example import Example
import json
from spacy.tokens import Span, Doc
from spacy_ner_utils import *


def evaluate(model_path, test_path, logger_file):
    nlp = spacy.load(model_path)
    gold_examples = load_data(nlp, test_path)

    scorer = Scorer()

    for example in gold_examples:
        example.predicted = nlp(example.reference.text)
        scorer.score(example)

    scores = scorer.score
    ner_f = scores["ents_f"]
    ner_p = scores["ents_p"]
    ner_r = scores["ents_r"]

    print(f"Precision: {ner_p:.4f}")
    print(f"Recall:    {ner_r:.4f}")
    print(f"F1-score:  {ner_f:.4f}")

    ner_log = {}
    ner_log["NER_model"] = {
            "precision": ner_p,
            "recall": ner_r,
            "f1": ner_f
    }

    with open(logger_file, "a") as f:
        f.write(json.dumps(ner_log) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--logger", required=True)
    args = parser.parse_args()

    evaluate(args.model, args.test, args.logger)
