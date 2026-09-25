# Synthetic Fine-Tuning for Multilingual Biomedical Named Entity Recognition

This repository implements a synthetic-data generation and training pipeline for biomedical named entity recognition (BioNER) in English, Spanish, German, and French. It addresses a central problem in BioNER: strong supervised models normally require large, expensive, expert-annotated datasets.

The pipeline samples disease terms from biomedical terminologies and uses a decoder language model to generate and annotate synthetic text. The synthetic examples are used to pretrain a smaller BERT-based BioNER model with a bidirectional recurrent layer, multi-head attention (MHA), and a conditional random field (CRF). The model is then fine-tuned on 1%, 5%, 10%, 20%, 50%, or 100% of a gold-labelled downstream dataset.

> **Repository status:** This research artifact is being prepared for release. Paths and cluster scripts still require local configuration. Items that are not yet available are marked `TODO` throughout this document.

## Paper

**Title:** TODO: add the final paper title  
**Authors:** TODO: add the complete author list  
**Venue/year:** TODO: add the venue and publication year  
**Paper:** TODO: add the DOI, publisher page, or preprint link

The accompanying paper studies whether synthetic fine-tuning can reduce the amount of manually annotated data required for multilingual BioNER. It compares zero-shot and three-shot synthetic generation on NCBI Disease (English), DisTEMIST (Spanish), BRONCO150 (German), and QUAERO (French), then evaluates the resulting BioNER models using several percentages of the gold training data. The largest improvements are observed in the lowest-resource settings.

The code associated with the paper is maintained on the [`paper` branch](https://github.com/VidasMarta/bioNER/tree/paper).

## Main results

The paper reports the following absolute F1-score improvements over baseline models trained only on the same gold-labelled subset:

| Gold training data | English (NCBI) | Spanish (DisTEMIST) | German (BRONCO150) | French (QUAERO) |
|---:|---:|---:|---:|---:|
| 1% | +45.0 | +19.4 | +14.9 | +15.8 |
| 5% | +10.2 | +7.8 | +4.1 | +6.4 |
| 10% | +4.4 | +3.1 | +4.5 | +3.9 |

> **TODO — main-results figure:** Add the paper figure containing the complete F1 results, including the 20%, 50%, and 100% gold-data settings.

## Method overview

1. Download the gold corpora and biomedical terminology resources.
2. Convert each corpus to the repository's sentence-level JSON format and create the 1%, 5%, 10%, 20%, and 50% training subsets.
3. Sample disease terms from SNOMED CT, ICD-10, and the Disease Ontology.
4. Generate zero-shot or three-shot synthetic sentences with a decoder language model served by llama.cpp.
5. Ask the language model to check or correct the entity annotations and apply language-specific spaCy processing.
6. Pretrain the BioNER model on the corrected synthetic data.
7. Fine-tune the model on a selected percentage of gold data.
8. Evaluate the model on the unchanged gold test set.

> **TODO — pipeline figure:** Add the methodology figure from the paper here.

The resulting dependency chain is:

```text
raw corpora + biomedical terminologies
                  |
                  v
        preprocessed gold JSON
          |                 |
          |                 +----> baseline BioNER training
          v
zero-/three-shot prompts + sampled terms
                  |
                  v
       raw synthetic JSONL
                  |
                  v
   corrected synthetic JSONL
                  |
                  v
       synthetic pretraining
                  |
                  v
       gold-data fine-tuning
                  |
                  v
       test logs + result CSVs
```

## Repository contents

```text
bioNER/
├── train.py                       # BioNER training and evaluation entry point
├── models.py                      # BiRNN, MHA, BERT, character CNN, and CRF model
├── datasets.py                    # Processed dataset loader
├── preprocessing.py               # Transformer tokenization and embeddings
├── evaluation.py                  # seqeval precision, recall, and F1 evaluation
├── hyper_param_tuning.py          # Optuna hyperparameter search
├── kshot_synthetic_generation.py  # Synthetic generation and correction pipeline
├── data_wranglig/                 # Corpus-specific preprocessing scripts
├── src_generate/                  # Prompt construction and generation utilities
├── experiments/
│   ├── data/                      # Raw-data preprocessing configurations
│   ├── ncbi/                      # English experiments
│   ├── distemist/                 # Spanish experiments
│   ├── bronco150/                 # German experiments
│   ├── quaero/                    # French experiments
│   ├── generate_kshot4B.sh        # 4B generation SLURM template
│   ├── generate_kshot27B.sh       # 27B generation SLURM template
│   └── preprocess.sh              # Preprocessing SLURM template
├── utils/                         # Trainers, logging, and output analysis
├── start-bioner-docker/           # Container recipe and Python requirements
└── settings.template.py           # Local path configuration template
```

Experiment directories contain three configuration families:

- `baseline/*.yml`: train only on a gold-data subset.
- `0shot/pretrain.yml` and `0shot/ft_*.yml`: pretrain on zero-shot synthetic data, then fine-tune on gold data.
- `3shot_<percentage>/pretrain.yml` and `3shot_<percentage>/ft_*.yml`: pretrain on three-shot synthetic data generated using examples from the indicated percentage of the gold training set, then fine-tune on gold data.

## Requirements

The supplied research workflow targets a Linux server or HPC cluster.

| Component | Requirement |
|---|---|
| Operating system | Linux; the supplied orchestration scripts use Bash and SLURM |
| Python | Python 3; TODO: record the exact tested minor version |
| Training runtime | PyTorch 2.4.1, CUDA 12.4, and cuDNN 9 in the supplied base image |
| GPU | NVIDIA CUDA GPU recommended for training; generation scripts request two GPUs |
| Generation runtime | Apptainer/Singularity, a CUDA llama.cpp server image, and local GGUF model weights |
| CPU and RAM | Generation jobs request 8 CPUs and 24 GB RAM; preprocessing requests 6 CPUs and 16 GB RAM |
| Storage | TODO: report raw-data, model, container, synthetic-data, and checkpoint sizes |

Python packages are listed in [`start-bioner-docker/requirements_base.txt`](start-bioner-docker/requirements_base.txt). The main dependencies include:

- PyTorch
- Transformers
- spaCy
- TorchCRF
- seqeval
- scikit-learn
- pandas
- NumPy
- PyYAML
- obonet
- requests
- Optuna

The current requirements file is not version-pinned. Exact reproduction may therefore require reconstructing and publishing the package versions used for the paper.

### Language models and encoders

The experiments expect local copies of the following encoder families beneath `EMBEDDINGS_PATH`:

| Experiment name | Expected local directory |
|---|---|
| `bioBERT` | `bioBERT_setup/` |
| `roBERTa_clinical_es` | `roBERTa_clinical_es_setup/` |
| `drBERT_fr` | `drBERT_fr_setup/` |
| `medBERT_ger` | `medBERT_ger_setup/` |

TODO: add the exact upstream model identifiers, revisions, licenses, and download or conversion commands used to create these directories.

The preprocessing and generation configurations additionally use these spaCy pipelines:

```text
en_core_web_trf
es_dep_news_trf
de_dep_news_trf
fr_dep_news_trf
```

## Installation and environment setup

Clone the paper branch and enter the repository:

```bash
git clone --branch paper https://github.com/VidasMarta/bioNER.git
cd bioNER
```

Create a Python environment and install the current dependency list:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r start-bioner-docker/requirements_base.txt
```

Install the language-specific spaCy pipelines. English can be installed directly with spaCy; the exact installation sources for the other transformer pipelines still need to be documented.

```bash
python -m spacy download en_core_web_trf

# TODO: install es_dep_news_trf
# TODO: install de_dep_news_trf
# TODO: install fr_dep_news_trf
```

Create the local settings file:

```bash
cp settings.template.py settings.py
```

Edit `settings.py` and configure all paths:

```python
MODEL_PATH = "/absolute/path/to/model-checkpoints"
DATA_PATH = "/absolute/path/to/processed-data"
OUTPUT_PATH = "/absolute/path/to/output"
EXPERIMENTS_PATH = "/absolute/path/to/bioNER/experiments"
LOG_PATH = "/absolute/path/to/logs"
EMBEDDINGS_PATH = "/absolute/path/to/local-transformer-models"
```

Create the model and log directories before training. `settings.py` is ignored by Git so machine-specific paths are not committed.

### Container setup

The intended image is based on:

```text
pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
```

The current Docker recipe is a development snapshot and is **not yet directly buildable**: its build-context assumptions must be corrected and the referenced `test_imports.py` file is absent.

The intended build interface is:

```bash
# TODO: this command must be validated after repairing the Dockerfile.
docker build -t syn-bioner -f start-bioner-docker/Dockerfile .
```

TODO:

- Repair and test the Dockerfile.
- Add `test_imports.py`.
- Install all four required spaCy pipelines.
- Document `docker run` with NVIDIA GPU support.
- Document the required data, model, output, and log volume mounts.

## Data

Gold corpora and terminology resources are not tracked in Git. Obtain each resource from its provider and comply with its individual terms.

| Language | Corpus | Source | Access notes |
|---|---|---|---|
| English | NCBI Disease | [NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) | Public download; verify the corpus terms before redistribution |
| Spanish | DisTEMIST | [Zenodo record 7614764](https://zenodo.org/records/7614764) | Public record; retain the accompanying license and citation |
| German | BRONCO150 | [BRONCO project page](https://www2.informatik.hu-berlin.de/~leser/bronco/index.html) | Not bundled; request access and sign the required data-use agreement |
| French | QUAERO | [DrBenchmark/QUAERO](https://huggingface.co/datasets/DrBenchmark/QUAERO) | Check the licenses of the EMEA and MEDLINE source material |

Synthetic prompts sample disease terminology from:

- SNOMED CT;
- language-specific ICD-10 resources; and
- the Disease Ontology snapshot obtained in August 2025.

These resources may have different access and redistribution conditions. In particular, do not redistribute restricted terminology or BRONCO150 content as part of a synthetic-data release without checking the governing agreements.

TODO: add the exact terminology releases, download URLs, checksums, license notes, and preprocessing commands.

### Expected data layout

The checked-in YAML files currently contain machine-specific absolute paths. After adapting them to the local machine, the intended layout is approximately:

```text
syn-bioner/
├── bioNER/
├── data/
│   ├── raw/
│   │   ├── MeSH_NCBI/
│   │   ├── distemist/
│   │   ├── BRONCO150/
│   │   └── quaero/
│   ├── KB/
│   │   ├── SNOMEDCT/
│   │   ├── ICD10_GE_CH/
│   │   ├── ICD10FR/
│   │   └── hetionet/
│   ├── processed/
│   │   ├── ncbi/trf/
│   │   ├── distemist/trf/
│   │   ├── bronco150/trf/
│   │   └── quaero/trf/
│   └── synthetic/
│       ├── ncbi/
│       ├── distemist/
│       ├── bronco150/
│       └── quaero/
├── models/
├── output/
└── logs/
```

## Reproducing the experiments

The commands below describe the shortest path through the current code. They are templates: review all YAML paths and shell variables before submitting a long-running job.

### 1. Preprocess a gold corpus

From the directory containing the `bioNER` package, run a corpus converter. For example, for NCBI Disease:

```bash
python3 -m bioNER.data_wranglig.ncbi_to_json \
  --config_file bioNER/experiments/data/ncbi.yml
```

Equivalent modules exist for the other corpora:

```bash
python3 -m bioNER.data_wranglig.distemist_to_json \
  --config_file bioNER/experiments/data/distemist.yml

python3 -m bioNER.data_wranglig.quaero_to_json \
  --config_file bioNER/experiments/data/quaero.yml

python3 -m bioNER.data_wranglig.bronco150_to_json \
  --config_file bioNER/experiments/data/bronco150.yml
```

On SLURM, edit `USERNAME`, `CORPORA`, `SPACY_MODELS`, and the container paths in `experiments/preprocess.sh`, then submit it:

```bash
sbatch bioNER/experiments/preprocess.sh
```

The converters create train, development, and test JSON files and percentage-based training subsets in the configured `parsed_mesh_folder`.

### 2. Generate synthetic data

The supplied generation scripts start a local llama.cpp server and invoke the generation client in an Apptainer/Singularity image.

Before submission, configure:

- `USERNAME`;
- `CORPUS` (`ncbi`, `distemist`, `quaero`, or `bronco150`);
- the GGUF `MODEL` and model directory;
- server and client container paths;
- scratch paths;
- batch size and port;
- zero-shot or three-shot settings; and
- annotation-checking settings.

Submit the appropriate model-size template:

```bash
sbatch bioNER/experiments/generate_kshot4B.sh
```

or:

```bash
sbatch bioNER/experiments/generate_kshot27B.sh
```

For a manually managed llama.cpp server, the Python client can be invoked directly from the repository root:

```bash
python3 -m bioNER.kshot_synthetic_generation \
  --config_file bioNER/experiments/ncbi/0shot_generation.yml
```

The configuration's `server_url` must expose a llama.cpp-compatible `POST /completion` endpoint.

A generation directory contains:

```text
generated_sentences_YYYYMMDD.jsonl            # Raw model text
corrected_generated_sentences_YYYYMMDD.jsonl  # Corrected annotations
term_list.txt                                 # Sampled terminology entries
timings.json                                  # Per-batch timings
generation.log                                # Generation diagnostics
```

### 3. Train a gold-only baseline

Training imports `settings.py` and should be launched from the `bioNER` directory:

```bash
cd bioNER

python train.py \
  --config experiments/ncbi/baseline/1pct.yml \
  --model_name NCBI_baseline_1pct
```

Each invocation trains and evaluates five seeded runs. Configurations for the other languages are located under their respective experiment directories.

### 4. Pretrain on synthetic data

The synthetic file named in the selected `pretrain.yml` must exist beneath `DATA_PATH`.

For example:

```bash
python train.py \
  --config experiments/ncbi/0shot/pretrain.yml \
  --model_name 27B_NCBI_0shot
```

The model name matters: downstream configurations use `pretrained_weights_path` to locate checkpoints with the following pattern:

```text
<model-name>_seed<seed>_best.bin
```

### 5. Fine-tune on gold data

After synthetic pretraining, fine-tune the model on the chosen gold percentage:

```bash
python train.py \
  --config experiments/ncbi/0shot/ft_1pct.yml \
  --model_name 27B_NCBI_0shot_ft_1pct
```

Repeat with the corresponding:

```text
ft_5pct.yml
ft_10pct.yml
ft_20pct.yml
ft_50pct.yml
ft_100pct.yml
```

Repeat the baseline, pretraining, and fine-tuning grid for all four corpora and for the zero-shot and three-shot synthetic conditions.

> **Known configuration gap:** The tracked NCBI `baseline/` directory currently contains only `1pct.yml` and `5pct.yml`. TODO: add the 10%, 20%, 50%, and 100% NCBI baseline configurations needed for the complete paper grid.

## Outputs and correspondence to the paper

| Repository output | Meaning | Paper use |
|---|---|---|
| `generated_sentences_*.jsonl` | Raw language-model generations | Intermediate generation artifact |
| `corrected_generated_sentences_*.jsonl` | Synthetic sentences with checked entity spans | Synthetic-pretraining input |
| `term_list.txt` | Sampled ontology or terminology entries | Generation provenance and coverage |
| `timings.json` | Generation and correction timings | Computational-cost analysis |
| `MODEL_PATH/*_best.bin` | Best validation checkpoint for each seed | Baseline, synthetic-pretrained, or fine-tuned model |
| `LOG_PATH/<model>/train.log` | Per-epoch training loss | Training diagnostics |
| `LOG_PATH/<model>/valid.log` | Validation loss and F1 | Model selection and early stopping |
| `LOG_PATH/<model>/test.log` | Per-seed test F1, precision, recall, and runtime | Reported evaluation inputs |
| `LOG_PATH/<configured-path>/test.csv` | Mean and standard deviation across seeds | Paper tables and plots |

Strict and non-strict micro-averaged scores are computed with `seqeval`. The aggregate CSV currently records non-strict F1, precision, and recall across the five seeds.

**Synthetic-data archive:** TODO: add the Zenodo or IEEE DataPort DOI and map each archive directory to its corpus, generator size, shot condition, checking condition, and gold-example percentage.

**Results archive:** TODO: publish the result CSV files and provide a table mapping every paper figure and table to an exact configuration, checkpoint, and log file.

## Generated versus provided artifacts

| Artifact | Provided in Git? | How it is obtained |
|---|---:|---|
| Source code and experiment YAML files | Yes | Clone this repository |
| Raw gold corpora | No | Download or request them from the corpus owner |
| Terminology resources | No | Obtain them from SNOMED CT, ICD-10 providers, and Disease Ontology |
| Processed gold JSON and percentage subsets | No | Run the `data_wranglig` converters |
| GGUF generator weights | No | TODO: document the exact model release and conversion |
| Local BERT-family encoders | No | TODO: document model identifiers and download steps |
| Raw synthetic generations | No | Run the generation scripts |
| Corrected synthetic datasets | No | Produced during generation; archive link TODO |
| Model checkpoints | No | Run baseline, pretraining, and fine-tuning experiments |
| Final result CSV files and plots | No | Aggregate the five-seed training logs; archive link TODO |

## Runtime and computational cost

A full reproduction is expected to take approximately **seven days on a server with two NVIDIA A100 GPUs**, although runtime depends strongly on generator size, GPU memory, llama.cpp build, batch size, storage performance, and queue time.

The included generation SLURM templates request:

- two GPUs;
- 24 GB system memory;
- eight CPU cores; and
- up to approximately five days for one submitted job.

Training runs each configuration five times.

TODO: report measured generation and training time, peak GPU memory, energy usage, and storage requirements for every model and corpus.

## Reproducibility and expected variation

Training uses the following seeds:

```text
42
198
6000
3828
7382
```

Synthetic generation uses seed `42`.

PyTorch deterministic cuDNN behavior is requested during training, but exact equality is not guaranteed across:

- GPU models;
- CUDA and cuDNN versions;
- PyTorch versions;
- tokenizer and model revisions;
- spaCy pipeline versions; or
- llama.cpp builds.

Other known limitations include:

- Requirements and model revisions are not pinned.
- Configuration files contain local absolute paths and dated synthetic filenames.
- Generation relies on external model weights and a separately built llama.cpp server container.
- Multilingual spaCy pipelines are not fully installed by the current Dockerfile.
- BRONCO150 requires approval and a data-use agreement.
- Preprocessing and annotation policies differ between the four corpora.
- LLM generation and GPU kernels can introduce numerical or textual variation.
- The Docker recipe and end-to-end commands have not yet been validated on a clean machine.
- Some experiment configurations use `bronco`, while others use `bronco150`; normalize these paths before running the German experiments.
- Different GPU hardware may change generation throughput, supported batch sizes, and generated output.

## License and reuse

**Software license:** TODO: choose a license and add a root `LICENSE` file. Until then, the repository does not state permission to copy, modify, or redistribute the software.

**Synthetic-data license:** TODO: add the selected open-data license to the data archive and clarify which source-data or terminology restrictions carry over.

Gold corpora, pretrained models, spaCy pipelines, biomedical terminologies, and container base images retain their own licenses and access conditions.

## Citation

TODO: add `CITATION.cff` after the paper metadata is final.

In the meantime, replace the placeholders below with the published citation:

```bibtex
@article{TODO,
  title   = {TODO: Paper title},
  author  = {TODO: Authors},
  journal = {TODO: Venue},
  year    = {TODO: Year},
  doi     = {TODO: DOI},
  url     = {TODO: Publication URL}
}
```

When using an individual gold corpus, terminology, language model, or pretrained encoder, cite that resource separately according to its provider's instructions.

## Contact

**Maintainer:** TODO: add name and email address.

For code defects or documentation problems, open an issue in the [GitHub repository](https://github.com/VidasMarta/bioNER/issues).

For restricted data access, contact the corresponding dataset owner rather than attaching data to a public issue.