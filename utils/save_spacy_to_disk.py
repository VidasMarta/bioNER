import spacy
import os
import argparse

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Save a spaCy model to disk")
parser.add_argument("--model", required=True, help="spaCy model name (e.g. de_dep_news_trf)")
parser.add_argument("--output", default="/models", help="Output directory (default: /models)")

args = parser.parse_args()

model_name = args.model
save_path = os.path.join(args.output, model_name)

# Ensure directory exists
os.makedirs(save_path, exist_ok=True)

# Load model
nlp = spacy.load(model_name)

# Save model
nlp.to_disk(save_path)

print(f"Model '{model_name}' saved to: {save_path}")