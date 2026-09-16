"""
GPU batch de-identification pipeline runner.

Runs the full pipeline — transformer NER (GPU) -> recognition (CPU: regex +
cached transformer results + known patient values) -> anonymization (CPU:
HIPS) — over a directory of input Parquet files, writing anonymized output
as Parquet.

Input schema (one row per note): text_hash, note_text, patient_uid
    (optional: patient_identifiers - JSON object of known PHI values,
     jitter - per-note date jitter override,
     recognizer_results_json - pre-computed NER spans, used only with
     --no-run-transformer to skip GPU inference)

Usage:
    python -m tide2.runner.pipeline \
        --input ./data/input --output ./data/output \\
        --model StanfordAIMI/stanford-deidentifier-v2 \\
        --salt-hex 00000000000000000000000000000000000000000000000000000000000000 \\
        --key-hex  1111111111111111111111111111111111111111111111111111111111111111
"""

import argparse
import concurrent.futures
import hashlib
import json
import logging
import multiprocessing
import os
from collections.abc import Iterator
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer import EntityRecognizer
from presidio_analyzer import RecognizerRegistry
from presidio_analyzer import RecognizerResult as AnalyzerRecognizerResult
from presidio_analyzer.context_aware_enhancers import ContextAwareEnhancer
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.predefined_recognizers import DateRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities import RecognizerResult

from tide2.anonymizers import AccessionNumberHashAnonymizer
from tide2.anonymizers import AgeGroupAnonymizer
from tide2.anonymizers import DateJitterAnonymizer
from tide2.anonymizers import FakerAnonymizer
from tide2.anonymizers import HipsAlphaNumericAnonymizer
from tide2.anonymizers import HipsLocationAnonymizer
from tide2.anonymizers import HipsNamesAnonymizer
from tide2.anonymizers import presidio_patches
from tide2.recognizers import AccessionRecognizer
from tide2.recognizers import AddressRecognizer
from tide2.recognizers import Base64ImageRecognizer
from tide2.recognizers import EmailRecognizer
from tide2.recognizers import GeneticSequenceRecognizer
from tide2.recognizers import HarRecognizer
from tide2.recognizers import MrnRecognizer
from tide2.recognizers import PhoneRecognizer
from tide2.recognizers import SsnRecognizer
from tide2.recognizers import UrlRecognizer
from tide2.recognizers import create_cached_recognizer
from tide2.recognizers import create_recognizers_for_patient
from tide2.recognizers.nlp_engine import _BlankSpacyNlpEngine
from tide2.transformers import TransformerCore
from tide2.transformers.reassembly import chunk_document_row
from tide2.transformers.reassembly import reassemble_chunks_for_document
from tide2.utils.span_metrics import resolve_recognizer_results
from tide2.utils.text_processing import aggregate_bio_tokens

logger = logging.getLogger(__name__)

# analyzer.analyze(entities=None) only auto-expands to entities supported by the
# static registry - it silently drops entity types that only ad_hoc_recognizers
# (the cached transformer results + per-patient known values) produce, such as
# PERSON/DOCTOR/HOSPITAL. Must be passed explicitly to analyze() below.
ALL_SUPPORTED_ENTITIES = [
    # Transformer entity types
    "PATIENT",
    "DOCTOR",
    "HCW",
    "PERSON",
    "HOSPITAL",
    "LOCATION",
    "DATE",
    "AGE",
    "ID",
    "PHONE",
    "WEB",
    "OTHER",
    "VENDOR",
    # Regex recognizer entity types
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "URL",
    "MRN",
    "HAR",
    "ACC_NUM",
    "CSN_ID",
    "BASE64_IMAGE",
    "GENETIC_SEQUENCE",
    # Known-values recognizer types
    "MEDICAL_LICENSE",
]


class NoOpContextEnhancer(ContextAwareEnhancer):
    """No-op context enhancer that returns results unchanged for maximum batch throughput."""

    def __init__(self) -> None:
        """Initialize with dummy parameters since we won't use them."""
        super().__init__(
            context_similarity_factor=0.0,
            min_score_with_context_similarity=0.0,
            context_prefix_count=0,
            context_suffix_count=0,
        )

    def enhance_using_context(
        self,
        text: str,
        raw_results: list[AnalyzerRecognizerResult],
        nlp_artifacts: NlpArtifacts,
        recognizers: list[EntityRecognizer],
        context: list[str] | None = None,
    ) -> list[AnalyzerRecognizerResult]:
        """Return results unchanged without any context enhancement, skipping the parent class's context similarity computations."""
        return raw_results


def infer_with_oom_retry(core: TransformerCore, texts: list[str], batch_size: int) -> list[list[dict]]:
    """Run GPU inference, halving the batch on CUDA OOM until it fits.

    TransformerCore.infer_raw_direct has no built-in OOM recovery, so an
    orchestration layer running on real hardware needs to provide it.
    """
    if not texts:
        return []
    try:
        return core.infer_raw_direct(texts, batch_size=min(batch_size, len(texts)))
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() or len(texts) == 1:
            raise
        logger.warning("CUDA OOM on batch of %d texts; splitting in half and retrying", len(texts))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = len(texts) // 2
        return infer_with_oom_retry(core, texts[:mid], batch_size) + infer_with_oom_retry(core, texts[mid:], batch_size)


def run_transformer_stage(
    core: TransformerCore,
    notes: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    gpu_batch_size: int,
) -> dict[str, str]:
    """Chunk notes, run GPU inference, and reassemble to recognizer_results_json per text_hash."""
    chunk_rows = [c for note in notes for c in chunk_document_row(note, chunk_size, chunk_overlap)]

    chunk_texts = [c["chunk_text"] for c in chunk_rows]

    # Sort by length before batching so each GPU batch pads to a similar length
    # instead of the longest chunk in an arbitrary (document-order) grouping -
    # padding waste was already shown to hurt (see DEV_TESTING.md hypothesis #3).
    # Character length is a cheap proxy for token count, good enough for bucketing.
    order = sorted(range(len(chunk_texts)), key=lambda i: len(chunk_texts[i]))
    sorted_texts = [chunk_texts[i] for i in order]
    sorted_predictions = infer_with_oom_retry(core, sorted_texts, gpu_batch_size)
    predictions_by_index = dict(zip(order, sorted_predictions, strict=True))
    raw_predictions = [predictions_by_index[i] for i in range(len(chunk_texts))]

    # reassemble_chunks_for_document expects aggregated entities (entity_group
    # key), not the raw per-token BIO predictions infer_raw_direct returns.
    for chunk_row, preds in zip(chunk_rows, raw_predictions, strict=True):
        aggregated = aggregate_bio_tokens(preds, chunk_row["chunk_text"])
        chunk_row["predictions_json"] = json.dumps(aggregated)

    chunks_by_doc: dict[str, list[dict]] = {}
    for chunk_row in chunk_rows:
        chunks_by_doc.setdefault(chunk_row["text_hash"], []).append(chunk_row)

    note_text_by_hash = {n["text_hash"]: (n.get("note_text") or "") for n in notes}
    return {
        text_hash: reassemble_chunks_for_document(rows, note_text_by_hash[text_hash], core.model_name)[0]
        for text_hash, rows in chunks_by_doc.items()
    }


def build_analyzer() -> AnalyzerEngine:
    """Assemble the regex + known-values recognizer registry (transformer results are ad-hoc, per-note)."""
    registry = RecognizerRegistry()
    registry.add_recognizer(DateRecognizer())
    registry.add_recognizer(EmailRecognizer())
    registry.add_recognizer(PhoneRecognizer())
    registry.add_recognizer(SsnRecognizer())
    registry.add_recognizer(UrlRecognizer())
    registry.add_recognizer(MrnRecognizer())
    registry.add_recognizer(HarRecognizer())
    registry.add_recognizer(AccessionRecognizer())
    registry.add_recognizer(Base64ImageRecognizer())
    registry.add_recognizer(GeneticSequenceRecognizer())
    registry.add_recognizer(AddressRecognizer())
    registry.remove_recognizer("SpacyRecognizer")

    import spacy

    blank_nlp = spacy.blank("en")
    blank_nlp.max_length = 2_000_000
    nlp_engine = _BlankSpacyNlpEngine(loaded_spacy_model=blank_nlp)

    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["en"],
        context_aware_enhancer=NoOpContextEnhancer(),
    )


def build_anonymizer() -> AnonymizerEngine:
    engine = AnonymizerEngine()
    for anonymizer in (
        AccessionNumberHashAnonymizer,
        FakerAnonymizer,
        DateJitterAnonymizer,
        HipsNamesAnonymizer,
        HipsAlphaNumericAnonymizer,
        HipsLocationAnonymizer,
        AgeGroupAnonymizer,
    ):
        engine.add_anonymizer(anonymizer)
    return engine


def build_operators(
    salt: bytes, key: bytes, acc_num_salt: str, acc_num_study_id: str, patient_uid: str, jitter: int
) -> dict[str, OperatorConfig]:
    """Per-note operator config — ACC_NUM and date jitter vary per patient/note."""
    return {
        "DEFAULT": OperatorConfig("redact"),
        "OTHER": OperatorConfig("redact"),
        "BASE64_IMAGE": OperatorConfig("redact"),
        "GENETIC_SEQUENCE": OperatorConfig("faker_anonymizer"),
        "AGE": OperatorConfig("age_grouping", {"upper_limit": 89}),
        "EMAIL_ADDRESS": OperatorConfig("faker_anonymizer"),
        "WEB": OperatorConfig("faker_anonymizer"),
        "URL": OperatorConfig("faker_anonymizer"),
        "PHONE_NUMBER": OperatorConfig("hips_alphanumeric", {"salt": salt, "key": key}),
        "PHONE": OperatorConfig("hips_alphanumeric", {"salt": salt, "key": key}),
        "ORGANIZATION": OperatorConfig("faker_anonymizer"),
        "VENDOR": OperatorConfig("hips_location", {"salt": salt, "key": key}),
        "US_SSN": OperatorConfig("hips_alphanumeric", {"salt": salt, "key": key}),
        "ZIP_CODE": OperatorConfig("hips_location", {"salt": salt, "key": key}),
        "LOCATION": OperatorConfig("hips_location", {"salt": salt, "key": key}),
        "HOSPITAL": OperatorConfig("hips_location", {"salt": salt, "key": key}),
        "MEDICAL_LICENSE": OperatorConfig("hips_alphanumeric", {"salt": salt, "key": key}),
        "PERSON": OperatorConfig("hips_names", {"salt": salt, "key": key}),
        "DOCTOR": OperatorConfig("hips_names", {"salt": salt, "key": key}),
        "PATIENT": OperatorConfig("hips_names", {"salt": salt, "key": key}),
        "HCW": OperatorConfig("hips_names", {"salt": salt, "key": key}),
        "MRN": OperatorConfig("hips_alphanumeric", {"salt": salt, "key": key}),
        "HAR": OperatorConfig("hips_alphanumeric", {"salt": salt, "key": key}),
        "ACC_NUM": OperatorConfig(
            "accession_number_hash",
            {"salt": acc_num_salt, "study_id": acc_num_study_id, "entity_type": patient_uid},
        ),
        "ID": OperatorConfig("hips_alphanumeric", {"salt": salt, "key": key}),
        "CSN_ID": OperatorConfig("hips_alphanumeric", {"salt": salt, "key": key}),
        "DATE_TIME": OperatorConfig("date_jitter", {"jitter": jitter}),
        "DATE": OperatorConfig("date_jitter", {"jitter": jitter}),
    }


def iter_note_batches(input_files: list[Path], row_batch_size: int) -> Iterator[list[dict]]:
    """Yield each row-batch of notes across all input parquet files."""
    for input_path in input_files:
        logger.info("Processing %s", input_path)
        parquet_file = pq.ParquetFile(input_path)
        for record_batch in parquet_file.iter_batches(batch_size=row_batch_size):
            yield record_batch.to_pandas().to_dict("records")


# Populated once per CPU worker process by _init_cpu_worker, not per task: AnalyzerEngine/
# AnonymizerEngine are expensive to construct and aren't picklable across the
# ProcessPoolExecutor boundary, so each worker builds and reuses its own copy.
# Single-element lists are appended to in place (rather than module-level
# `analyzer`/`anonymizer_engine` names rebound with `global`) to keep ruff's
# no-global-state rule happy while preserving each engine's concrete type.
_worker_analyzer: list[AnalyzerEngine] = []
_worker_anonymizer_engine: list[AnonymizerEngine] = []


def _init_cpu_worker() -> None:
    """ProcessPoolExecutor initializer: build this worker's analyzer/anonymizer once."""
    presidio_patches.disable_whitespace_merging()
    presidio_patches.patch_conflict_resolution()
    _worker_analyzer.append(build_analyzer())
    _worker_anonymizer_engine.append(build_anonymizer())


def run_cpu_stage(
    notes: list[dict],
    transformer_results_by_hash: dict[str, str],
    args: argparse.Namespace,
) -> list[dict]:
    """Run the recognizer -> anonymizer stages for one batch of notes.

    Runs inside a CPU worker process (see _init_cpu_worker): the regex
    recognizers spend most of their time in CPython's `re` engine, which does
    not release the GIL, so this stage runs in a separate process to overlap
    with GPU inference in the main process.
    """
    analyzer = _worker_analyzer[0]
    anonymizer_engine = _worker_anonymizer_engine[0]
    salt = bytes.fromhex(args.salt_hex)
    key = bytes.fromhex(args.key_hex)

    output_rows = []
    for note in notes:
        text_hash = note["text_hash"]
        note_text = note.get("note_text") or ""
        # None (not "") when absent, so a missing patient_uid hashes as the literal string "None" below
        patient_uid = note.get("patient_uid")
        patient_uid_str = patient_uid or ""
        patient_identifiers = json.loads(note.get("patient_identifiers") or "{}")

        cached_recognizer = create_cached_recognizer(results=transformer_results_by_hash.get(text_hash, "[]"))
        ad_hoc_recognizers: list[EntityRecognizer] = [
            cached_recognizer,
            *create_recognizers_for_patient(patient_identifiers),
        ]

        recognizer_results = analyzer.analyze(
            text=note_text, language="en", entities=ALL_SUPPORTED_ENTITIES, ad_hoc_recognizers=ad_hoc_recognizers
        )

        operators = build_operators(
            salt, key, args.acc_num_salt, args.acc_num_study_id, patient_uid_str, jitter=note.get("jitter", 30)
        )
        # presidio-anonymizer defines its own RecognizerResult distinct from presidio-analyzer's;
        # convert explicitly rather than relying on structural compatibility.
        anonymizer_results = [
            RecognizerResult(entity_type=r.entity_type, start=r.start, end=r.end, score=r.score)
            for r in recognizer_results
        ]
        # patch_conflict_resolution() disabled Presidio's own O(n^2) resolution
        # (see presidio_patches docstring), so overlaps across recognizers must
        # be resolved explicitly before anonymizing.
        anonymizer_results = resolve_recognizer_results(anonymizer_results)
        anonymized = anonymizer_engine.anonymize(
            text=note_text, analyzer_results=anonymizer_results, operators=operators
        )

        # Stable row identifier for downstream joins/checkpointing.
        row_id_key = f"{text_hash}:{patient_uid if patient_uid is not None else 'None'}"
        row_id = hashlib.sha256(row_id_key.encode()).hexdigest()
        output_rows.append(
            {
                "text_hash": text_hash,
                "patient_uid": patient_uid,
                "anonymized_note_text": anonymized.text,
                "anonymizer_results_json": json.dumps(
                    [
                        {
                            "start": item.start,
                            "end": item.end,
                            "entity_type": item.entity_type,
                            "text": item.text,
                            "operator": item.operator,
                        }
                        for item in anonymized.items
                    ]
                ),
                "entity_count": len(anonymized.items),
                "processing_status": "success",
                "error_message": None,
                "row_id": row_id,
                "processing_timestamp": datetime.now(UTC).isoformat(),
            }
        )

    return output_rows


def run_pipeline(input_files: list[Path], output_path: Path, args: argparse.Namespace, device: str) -> int:
    """Run the pipeline over one shard of input files, writing one output parquet.

    Loads its own `TransformerCore` (unless `args.run_transformer` is False), so
    this is safe to call from independent worker processes each targeting the
    same GPU (see `--num-gpu-workers`): each worker only uses ~1.2GB of model
    weights, and any single worker's forward pass leaves the GPU's compute
    mostly idle while it's busy on the CPU-bound recognizer/anonymizer stage,
    so multiple workers' GPU calls can genuinely interleave instead of just
    taking turns.
    """
    core: TransformerCore | None = None
    if args.run_transformer:
        core = TransformerCore(model_name=args.model, device=device, load_immediately=True)
        logger.info("Loaded %s on %s (pid %d)", args.model, core.get_device_info(), os.getpid())
    else:
        logger.info("--no-run-transformer: reusing recognizer_results_json from --input (pid %d)", os.getpid())

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    def flush(pending: concurrent.futures.Future | None) -> None:
        nonlocal writer, total_rows
        if pending is None:
            return
        out_table = pa.Table.from_pylist(pending.result())
        if writer is None:
            writer = pq.ParquetWriter(output_path, out_table.schema)
        writer.write_table(out_table)
        total_rows += out_table.num_rows
        logger.info("  wrote %d rows (%d total)", out_table.num_rows, total_rows)

    # run_cpu_stage is regex- and pure-Python-heavy and holds the GIL almost
    # continuously, so it runs in a separate process: this process keeps driving
    # GPU inference for batch N while the worker process runs the CPU stage for
    # batch N-1.
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, initializer=_init_cpu_worker) as cpu_executor:
            pending: concurrent.futures.Future | None = None
            for notes in iter_note_batches(input_files, args.row_batch_size):
                if core is not None:
                    transformer_results_by_hash = run_transformer_stage(
                        core, notes, args.chunk_size, args.chunk_overlap, args.gpu_batch_size
                    )
                else:
                    transformer_results_by_hash = {
                        note["text_hash"]: note.get("recognizer_results_json") or "[]" for note in notes
                    }
                flush(pending)
                pending = cpu_executor.submit(run_cpu_stage, notes, transformer_results_by_hash, args)
            flush(pending)
    finally:
        if writer is not None:
            writer.close()

    logger.info("Done: %d notes -> %s", total_rows, output_path)
    return total_rows


def _shard(items: list[Path], num_shards: int) -> list[list[Path]]:
    """Split `items` into `num_shards` round-robin, roughly equal groups (empty groups dropped)."""
    shards = [items[i::num_shards] for i in range(num_shards)]
    return [s for s in shards if s]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Directory of input .parquet files (one row per note)")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory - one partition parquet file per GPU worker",
    )
    parser.add_argument("--model", default="StanfordAIMI/stanford-deidentifier-v2")
    parser.add_argument(
        "--run-transformer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the GPU transformer NER stage. With --no-run-transformer, skip it entirely and "
            "reuse each row's existing recognizer_results_json column from --input instead "
            "(defaults to '[]' for rows without one) - for re-running the recognizer/anonymizer "
            "stages against already-computed NER predictions without paying for GPU inference again."
        ),
    )
    parser.add_argument("--salt-hex", required=True, help="64 hex chars (32 bytes)")
    parser.add_argument("--key-hex", required=True, help="64 hex chars (32 bytes)")
    parser.add_argument("--acc-num-salt", default="")
    parser.add_argument("--acc-num-study-id", default="")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens for long notes")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Chunk overlap in tokens")
    parser.add_argument("--gpu-batch-size", type=int, default=64, help="Max chunks per GPU forward pass")
    parser.add_argument(
        "--row-batch-size",
        type=int,
        default=512,
        help=(
            "Notes read per input row-batch. This is also the candidate pool that "
            "run_transformer_stage's length-bucketing sorts before slicing into "
            "--gpu-batch-size chunks, so bigger pools bucket more precisely (see "
            "DEV_TESTING.md hypothesis #11) - unlike --gpu-batch-size, going bigger "
            "here doesn't reintroduce padding waste since this isn't padded as a "
            "single unit, so it's safe to raise further for large corpora."
        ),
    )
    parser.add_argument(
        "--num-gpu-workers",
        type=int,
        default=3,
        help=(
            "Number of independent worker processes, each loading its own model onto the "
            "same GPU and processing a shard of --input's files concurrently. A single "
            "worker's GPU calls leave the GPU's compute idle much of the time (it's waiting "
            "on the CPU-bound recognizer/anonymizer stage) while using very little of its "
            "memory, so multiple workers can share the GPU productively instead of each "
            "one taking turns. Each worker always writes its own worker{N}.parquet inside "
            "--output, even when this is 1 - --output is always a directory."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    input_files = sorted(Path(args.input).glob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No .parquet files found in {args.input}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_gpu_workers <= 1:
        run_pipeline(input_files, output_dir / "worker0.parquet", args, device)
        return

    shards = _shard(input_files, args.num_gpu_workers)
    logger.info("Sharding %d input files across %d GPU workers -> %s/", len(input_files), len(shards), output_dir)

    # CUDA is unsafe after fork() once a context has been initialized, and each worker
    # initializes its own context (via TransformerCore) - "spawn" avoids that entirely
    # by starting fresh interpreters instead of forking this (CUDA-free) parent.
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=run_pipeline, args=(shard, output_dir / f"worker{i}.parquet", args, device))
        for i, shard in enumerate(shards)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    failed = [p.pid for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"GPU worker process(es) failed (pids {failed}); see logs above for the actual error")


if __name__ == "__main__":
    main()
