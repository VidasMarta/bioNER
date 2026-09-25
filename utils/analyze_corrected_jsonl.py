#!/usr/bin/env python3
"""Analyze corrected synthetic-generation JSONL files.

The script recursively scans selected dataset directories, chooses the largest
``*corrected*.jsonl`` file in each experiment directory, calculates term/token
statistics, and writes CSV/JSON summaries plus per-dataset PNG dashboards.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path("/home/mkeber/syn-bioner/data")
DEFAULT_DATASETS = ("bronco150", "distemist", "ncbi", "quaero")
MODEL_RE = re.compile(r"(?<![A-Za-z0-9])(4B|27B)(?![A-Za-z0-9])", re.IGNORECASE)
MODEL_ORDER = {"4B": 0, "27B": 1, "UNKNOWN": 2}

SUMMARY_FIELDS = (
    "dataset",
    "model",
    "experiment",
    "experiment_group",
    "source_directory",
    "source_file",
    "file_size_bytes",
    "valid_records",
    "malformed_lines",
    "base_term_occurrences",
    "unique_base_terms",
    "repeated_distinct_base_terms",
    "llm_span_terms",
    "additional_llm_terms",
    "total_spans",
    "records_without_annotations",
    "heuristic_span_terms",
    "tokens",
    "records_missing_base_terms",
    "records_missing_spans_llm",
    "records_missing_tokens_and_tags",
)

AGGREGATE_FIELDS = (
    "dataset",
    "model",
    "experiment_count",
    "valid_records",
    "malformed_lines",
    "base_term_occurrences",
    "unique_base_terms",
    "repeated_distinct_base_terms",
    "llm_span_terms",
    "additional_llm_terms",
    "total_spans",
    "records_without_annotations",
    "heuristic_span_terms",
    "tokens",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the largest corrected JSONL file in every experiment "
            "directory and compare 4B with 27B outputs."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Data tree to scan (default: {DEFAULT_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/syn_gen_stat"),
        help="Directory for CSV, JSON, and PNG outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset directory names to include.",
    )
    parser.add_argument(
        "--include-unknown-model",
        action="store_true",
        help="Include experiment paths that do not contain 4B or 27B.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write tabular reports only.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first malformed JSON line instead of reporting it.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def normalize_term(value: Any) -> str:
    """Return a Unicode-normalized, whitespace-collapsed, lowercase term."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.split()).casefold()


def dataset_from_path(path: Path, root: Path, datasets: Iterable[str]) -> str | None:
    wanted = {name.casefold(): name for name in datasets}
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        if part.casefold() in wanted:
            return wanted[part.casefold()]
    return None


def model_from_path(path: Path) -> str:
    matches = MODEL_RE.findall(str(path))
    if not matches:
        return "UNKNOWN"
    models = {match.upper() for match in matches}
    if len(models) > 1:
        logging.warning("Both 4B and 27B occur in path; using final match: %s", path)
    return matches[-1].upper()


def experiment_group(name: str) -> str:
    """Remove the model token so matching 4B/27B runs share a chart category."""
    grouped = MODEL_RE.sub("", name)
    grouped = re.sub(r"[_\-.]{2,}", "_", grouped).strip("_-. ")
    return grouped or name


def discover_files(root: Path, datasets: list[str]) -> tuple[list[Path], list[dict[str, Any]]]:
    """Select the largest corrected JSONL in each parent directory."""
    candidates: dict[Path, list[Path]] = defaultdict(list)
    for path in root.rglob("*.jsonl"):
        if "corrected" not in path.name.casefold():
            continue
        if dataset_from_path(path, root, datasets) is None:
            continue
        try:
            if path.is_file():
                candidates[path.parent].append(path)
        except OSError as exc:
            logging.warning("Cannot inspect %s: %s", path, exc)

    selected: list[Path] = []
    manifest: list[dict[str, Any]] = []
    for directory, paths in sorted(candidates.items(), key=lambda item: str(item[0])):
        sized: list[tuple[int, str, Path]] = []
        for path in paths:
            try:
                sized.append((path.stat().st_size, path.name, path))
            except OSError as exc:
                logging.warning("Cannot stat %s: %s", path, exc)
        if not sized:
            continue
        # Filename is a deterministic tie-breaker when byte sizes are equal.
        size, _, winner = max(sized, key=lambda item: (item[0], item[1]))
        selected.append(winner)
        for candidate_size, _, candidate in sorted(sized, key=lambda item: item[1]):
            manifest.append(
                {
                    "directory": str(directory),
                    "file": str(candidate),
                    "size_bytes": candidate_size,
                    "selected": candidate == winner,
                    "selection_reason": (
                        "largest corrected JSONL in directory"
                        if candidate == winner
                        else f"smaller than selected file {winner.name} ({size} bytes)"
                    ),
                }
            )
    return selected, manifest


def list_value(record: dict[str, Any], key: str) -> list[Any] | None:
    value = record.get(key)
    return value if isinstance(value, list) else None


def analyze_file(path: Path, root: Path, datasets: list[str], strict: bool) -> tuple[dict[str, Any], Counter[str]]:
    dataset = dataset_from_path(path, root, datasets)
    if dataset is None:
        raise ValueError(f"Could not determine dataset for {path}")

    counters = Counter()
    terms: Counter[str] = Counter()
    print('+'*100)
    print(f"Analyzing {path}...")
    print('+'*100)
    with path.open("r", encoding="utf-8") as handle:
        print(f"Analyzing {path}...")
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSON value is not an object")
            except (json.JSONDecodeError, ValueError) as exc:
                counters["malformed_lines"] += 1
                message = f"{path}:{line_number}: {exc}"
                if strict:
                    raise ValueError(message) from exc
                logging.warning("Skipping malformed line: %s", message)
                continue

            counters["valid_records"] += 1

            base_terms = list_value(record, "base_terms")
            if base_terms is None:
                counters["records_missing_base_terms"] += 1
                base_terms = []
            counters["base_term_occurrences"] += len(base_terms)
            for term in base_terms:
                normalized = normalize_term(term)
                if normalized:
                    terms[normalized] += 1

            spans_llm = list_value(record, "spans_llm")
            if spans_llm is None:
                counters["records_missing_spans_llm"] += 1
                spans_llm = []
            counters["llm_span_terms"] += len(spans_llm)
            for i in spans_llm:
                if len(i) > 1:
                    if i[2] in base_terms:
                        # print(f"found {i[2]} in base_terms")
                        # counters["heuristic_span_terms"] += 1
                        continue
                    else:
                        # print(f"{i[2]} NOT NOT  NOT !!!! in base_terms")
                        counters["additional_llm_terms"] += 1
            tokens = list_value(record, "tokens")
            if line_number <= 6:
                print("SPANS:", spans_llm)
                print("BASE_TERMS:", base_terms)
                print("TOKENS:", tokens)
                print('*'*40)
            if tokens is not None:
                counters["tokens"] += len(tokens)
            else:
                tags = list_value(record, "tags")
                if tags is not None:
                    counters["tokens"] += len(tags)
                else:
                    counters["records_missing_tokens_and_tags"] += 1
            spans= list_value(record, "spans")
            if len(spans):
                counters["total_spans"] += len(spans)
            else:
                counters["records_without_annotations"] += 1
            for i in spans:
                if len(i) > 1:
                    if i[2] in base_terms:
                        # print(f"found {i[2]} in base_terms")
                        counters["heuristic_span_terms"] += 1
            
    model = model_from_path(path.parent)
    row: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "experiment": path.parent.name,
        "experiment_group": experiment_group(path.parent.name),
        "source_directory": str(path.parent),
        "source_file": str(path),
        "file_size_bytes": path.stat().st_size,
        **{field: counters[field] for field in SUMMARY_FIELDS if field in counters},
    }
    row["unique_base_terms"] = len(terms)
    row["repeated_distinct_base_terms"] = sum(count > 1 for count in terms.values())
    for field in SUMMARY_FIELDS:
        row.setdefault(field, 0)
    return row, terms


def aggregate_rows(
    rows: list[dict[str, Any]],
    term_counters: list[Counter[str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], Counter[str]]]:
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_terms: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row, terms in zip(rows, term_counters):
        key = (str(row["dataset"]), str(row["model"]))
        grouped_rows[key].append(row)
        grouped_terms[key].update(terms)

    aggregate: list[dict[str, Any]] = []
    summed = (
        "valid_records",
        "malformed_lines",
        "base_term_occurrences",
        "llm_span_terms",
        "additional_llm_terms",
        "total_spans",
        "records_without_annotations",
        "tokens",
    )
    for key in sorted(grouped_rows, key=lambda item: (item[0], MODEL_ORDER.get(item[1], 99))):
        dataset, model = key
        group = grouped_rows[key]
        terms = grouped_terms[key]
        item: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "experiment_count": len(group),
            **{field: sum(int(row[field]) for row in group) for field in summed},
            "unique_base_terms": len(terms),
            "repeated_distinct_base_terms": sum(count > 1 for count in terms.values()),
        }
        aggregate.append(item)
    return aggregate, grouped_terms


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    fields = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    output_dir: Path,
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    term_counters: list[Counter[str]],
    grouped_terms: dict[tuple[str, str], Counter[str]],
    manifest: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "experiment_summary.csv", rows, SUMMARY_FIELDS)
    write_csv(output_dir / "dataset_model_summary.csv", aggregate, AGGREGATE_FIELDS)
    write_csv(
        output_dir / "file_selection_manifest.csv",
        manifest,
        ("directory", "file", "size_bytes", "selected", "selection_reason"),
    )

    term_rows: list[dict[str, Any]] = []
    for row, terms in zip(rows, term_counters):
        for term, frequency in terms.most_common():
            term_rows.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "experiment": row["experiment"],
                    "source_file": row["source_file"],
                    "normalized_term": term,
                    "frequency": frequency,
                    "repeats_more_than_once": frequency > 1,
                }
            )
    write_csv(
        output_dir / "experiment_term_frequencies.csv",
        term_rows,
        (
            "dataset",
            "model",
            "experiment",
            "source_file",
            "normalized_term",
            "frequency",
            "repeats_more_than_once",
        ),
    )

    aggregate_term_rows: list[dict[str, Any]] = []
    for (dataset, model), terms in sorted(grouped_terms.items()):
        for term, frequency in terms.most_common():
            aggregate_term_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "normalized_term": term,
                    "frequency": frequency,
                    "repeats_more_than_once": frequency > 1,
                }
            )
    write_csv(
        output_dir / "dataset_model_term_frequencies.csv",
        aggregate_term_rows,
        ("dataset", "model", "normalized_term", "frequency", "repeats_more_than_once"),
    )

    payload = {
        "definitions": {
            "normalized_term": "Unicode NFKC, collapsed whitespace, and casefolded text",
            "repeated_distinct_base_terms": "distinct normalized base terms with frequency > 1",
            "additional_llm_terms": "sum(max(0, len(spans_llm) - len(base_terms)))",
            "tokens": "len(tokens), falling back to len(tags) when tokens is absent",
        },
        "experiment_summary": rows,
        "dataset_model_summary": aggregate,
        "file_selection_manifest": manifest,
    }
    with (output_dir / "statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def create_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logging.warning("matplotlib/numpy unavailable; skipping PNG visualizations")
        return []

    metrics = (
        ("valid_records", "Valid records"),
        ("base_term_occurrences", "Base-term occurrences"),
        ("unique_base_terms", "Unique base terms selected from ontologies"),
        ("repeated_distinct_base_terms", "Repeated distinct terms"),
        ("llm_span_terms", "All LLM span terms includes heuristic and llm detected spans of terms"),
        ("additional_llm_terms", "Additional LLM self-check terms"),
        ("heuristic_span_terms", "Heuristic span terms"),
        ("total_spans", "Total spans with annotations"),
        ("records_without_annotations", "Records without annotations"),
        ("tokens", "Generated tokens"),
    )
    created: list[Path] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in datasets:
        data = [row for row in rows if row["dataset"] == dataset]
        groups = sorted({str(row["experiment_group"]) for row in data})
        if not groups:
            continue
        lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for row in data:
            key = (str(row["experiment_group"]), str(row["model"]))
            if key in lookup:
                logging.warning(
                    "Multiple directories map to chart group %s/%s/%s; summing values",
                    dataset,
                    key[0],
                    key[1],
                )
                combined = dict(lookup[key])
                for metric, _ in metrics:
                    combined[metric] = int(combined[metric]) + int(row[metric])
                lookup[key] = combined
            else:
                lookup[key] = row

        height = max(9.0, 0.55 * len(groups) + 5.0)
        fig, axes = plt.subplots(4, 2, figsize=(18, height), constrained_layout=True)
        y = np.arange(len(groups), dtype=float)
        models = [model for model in ("4B", "27B", "UNKNOWN") if any(
                    r["model"] == model for r in data)]
        bar_height = 0.8 / max(1, len(models))
        offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * bar_height

        for axis, (metric, title) in zip(axes.flat, metrics):
            for offset, model in zip(offsets, models):
                values = [int(lookup.get((group, model), {}).get(metric, 0)) for group in groups]
                axis.barh(y + offset, values, height=bar_height, label=model)
            axis.set_title(title)
            axis.set_yticks(y, groups)
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=0.25)
            axis.tick_params(axis="y", labelsize=8)
            axis.legend(title="Model")
        # axes.flat[-1].axis("off")
        fig.suptitle(f"{dataset}: corrected JSONL experiment statistics", fontsize=16)
        output_path = output_dir / f"{dataset}_experiment_model_statistics.png"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(output_path)
    return created


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        print(f"error: data root is not a directory: {root}", file=sys.stderr)
        return 2

    selected, manifest = discover_files(root, args.datasets)
    if not selected:
        print(
            f"error: no corrected JSONL files found under {root} for: "
            f"{', '.join(args.datasets)}",
            file=sys.stderr,
        )
        return 1

    rows: list[dict[str, Any]] = []
    term_counters: list[Counter[str]] = []
    for path in selected:
        model = model_from_path(path.parent)
        if model == "UNKNOWN" and not args.include_unknown_model:
            logging.warning("Skipping path without 4B/27B model marker: %s", path)
            continue
        try:
            row, terms = analyze_file(path, root, args.datasets, args.strict)
        except (OSError, ValueError) as exc:
            if args.strict:
                raise
            logging.error("Could not analyze %s: %s", path, exc)
            continue
        rows.append(row)
        term_counters.append(terms)

    if not rows:
        print("error: no selected files could be analyzed", file=sys.stderr)
        return 1

    rows_with_terms = sorted(
        zip(rows, term_counters),
        key=lambda item: (
            item[0]["dataset"],
            item[0]["experiment_group"],
            MODEL_ORDER.get(str(item[0]["model"]), 99),
        ),
    )
    rows = [item[0] for item in rows_with_terms]
    term_counters = [item[1] for item in rows_with_terms]
    aggregate, grouped_terms = aggregate_rows(rows, term_counters)
    write_reports(output_dir, rows, aggregate, term_counters, grouped_terms, manifest)
    plot_paths = [] if args.no_plots else create_plots(output_dir, rows)

    print(f"Analyzed {len(rows)} experiment directories.")
    print(f"Reports written to: {output_dir}")
    if not args.no_plots:
        print(f"PNG dashboards created: {len(plot_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
