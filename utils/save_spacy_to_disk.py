import spacy
import os
import argparse
if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Load or save a spaCy model")
    parser.add_argument("--model", required=True, help="spaCy model name (e.g. de_dep_news_trf)")
    parser.add_argument("--output", default="/models", help="Output directory (default: /models)")

    args = parser.parse_args()

    model_name = args.model
    save_path = os.path.join(args.output, model_name)

    if os.path.exists(save_path) and os.listdir(save_path):
        # Load from disk if already saved
        print(f"Loading model from disk: {save_path}")
        nlp = spacy.load(save_path)
    else:
        # Otherwise load (and download if needed) and save
        print(f"Loading model from spaCy: {model_name}")
        nlp = spacy.load(model_name)

        os.makedirs(save_path, exist_ok=True)
        nlp.to_disk(save_path)

        print(f"Model '{model_name}' saved to: {save_path}")