"""Import a local SwissDial dataset into the ignored runtime dataset directory."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import random
import re
import shutil
import zipfile


DEFAULT_CATALOG_SIZE = 160
DEFAULT_SAMPLE_SEED = 42
DIALECT_NAMES = {
    "ag": "Aargau German",
    "be": "Bernese German",
    "bs": "Basel German",
    "gr": "Grisons German",
    "lu": "Lucerne German",
    "sg": "St. Gallen German",
    "vs": "Valais German",
    "zh": "Zurich German",
}
AUDIO_FILE_PATTERN = re.compile(r"ch_(?P<dialect>[a-z]{2})_(?P<sentence_id>\d+)\.wav", re.IGNORECASE)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to an extracted ETH SwissDial directory or TSV ZIP export")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/swissdial"),
        help="Local dataset directory (default: data/swissdial)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing imported clip files")
    parser.add_argument(
        "--catalog-size",
        "--sample-size",
        dest="catalog_size",
        type=int,
        default=DEFAULT_CATALOG_SIZE,
        help=(
            "Number of clips made available to the app, balanced by dialect; "
            f"0 imports all (default: {DEFAULT_CATALOG_SIZE})"
        ),
    )
    parser.add_argument(
        "--dialects",
        nargs="+",
        choices=tuple(DIALECT_NAMES),
        default=list(DIALECT_NAMES),
        help="Dialect folders to include (default: all eight)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help=f"Deterministic sampling seed (default: {DEFAULT_SAMPLE_SEED})",
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help=(
            "Only (re)build examples.jsonl from the source directory's sentences_ch_de_numerics.json "
            "for the existing manifest; no audio is required"
        ),
    )
    return parser.parse_args()


def _write_manifest(output_dir: Path, records: list[dict[str, object]]) -> None:
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"\w+", text.casefold()) if len(word) > 2}


def build_example_pool(dataset_dir: Path, output_dir: Path, max_overlap: float = 0.5) -> int:
    """Write text-only dialect/High German pairs that cannot leak evaluated sentences.

    Excludes every sentence ID in the manifest (sentences are parallel across dialects) and any
    sentence whose High German text overlaps an evaluated sentence by more than ``max_overlap``
    (Jaccard over content words), which removes templated near-duplicates.
    """
    transcript_path = dataset_dir / "sentences_ch_de_numerics.json"
    if not transcript_path.is_file():
        raise ValueError(f"Official SwissDial transcript file not found: {transcript_path}")
    manifest_path = output_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"Import the catalog before building examples: {manifest_path} is missing")

    evaluated = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluated_ids = {int(record["sentence_id"]) for record in evaluated if record.get("sentence_id") is not None}
    evaluated_words = [
        _content_words(str(record.get("standard_german_transcript", "")))
        for record in evaluated
        if record.get("standard_german_transcript")
    ]

    records: list[dict[str, object]] = []
    for sentence in json.loads(transcript_path.read_text(encoding="utf-8")):
        if not isinstance(sentence, dict) or "id" not in sentence or int(sentence["id"]) in evaluated_ids:
            continue
        standard = str(sentence.get("de", "")).strip()
        words = _content_words(standard)
        if not standard or not words:
            continue
        if any(len(words & other) / len(words | other) > max_overlap for other in evaluated_words if other):
            continue
        dialects = {
            dialect: str(sentence.get(f"ch_{dialect}", "")).strip()
            for dialect in DIALECT_NAMES
            if str(sentence.get(f"ch_{dialect}", "")).strip()
        }
        if dialects:
            records.append({"sentence_id": int(sentence["id"]), "de": standard, "dialects": dialects})

    (output_dir / "examples.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return len(records)


def _balanced_sample(
    candidates_by_dialect: dict[str, list[tuple[Path, dict[str, object]]]],
    sample_size: int,
    seed: int,
) -> list[tuple[str, Path, dict[str, object]]]:
    if sample_size < 0:
        raise ValueError("Sample size must be zero or greater.")

    randomizer = random.Random(seed)
    shuffled = {dialect: sorted(candidates, key=lambda candidate: candidate[0].name) for dialect, candidates in candidates_by_dialect.items()}
    for candidates in shuffled.values():
        randomizer.shuffle(candidates)

    if sample_size == 0:
        sample_size = sum(len(candidates) for candidates in shuffled.values())

    selected: list[tuple[str, Path, dict[str, object]]] = []
    offsets = {dialect: 0 for dialect in shuffled}
    while len(selected) < sample_size:
        added = False
        for dialect, candidates in shuffled.items():
            offset = offsets[dialect]
            if offset >= len(candidates):
                continue
            audio_path, sentence = candidates[offset]
            selected.append((dialect, audio_path, sentence))
            offsets[dialect] += 1
            added = True
            if len(selected) == sample_size:
                break
        if not added:
            break
    return selected


def import_official_dataset(
    dataset_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    catalog_size: int = DEFAULT_CATALOG_SIZE,
    dialects: list[str] | None = None,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> int:
    """Import a balanced sample from an extracted official ETH SwissDial 1.1 dataset."""
    transcript_path = dataset_dir / "sentences_ch_de_numerics.json"
    if not transcript_path.is_file():
        raise ValueError(f"Official SwissDial transcript file not found: {transcript_path}")

    raw_sentences = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(raw_sentences, list):
        raise ValueError(f"Official SwissDial transcript file must contain a JSON array: {transcript_path}")
    sentences = {
        int(sentence["id"]): sentence
        for sentence in raw_sentences
        if isinstance(sentence, dict) and "id" in sentence
    }

    selected_dialects = dialects or list(DIALECT_NAMES)
    candidates_by_dialect: dict[str, list[tuple[Path, dict[str, object]]]] = {}
    for dialect in selected_dialects:
        dialect = dialect.lower()
        if dialect not in DIALECT_NAMES:
            raise ValueError(f"Unknown SwissDial dialect: {dialect}")
        dialect_dir = dataset_dir / dialect
        if not dialect_dir.is_dir():
            raise ValueError(f"SwissDial dialect directory not found: {dialect_dir}")

        transcript_key = f"ch_{dialect}"
        candidates: list[tuple[Path, dict[str, object]]] = []
        for audio_path in dialect_dir.glob(f"ch_{dialect}_*.wav"):
            match = AUDIO_FILE_PATTERN.fullmatch(audio_path.name)
            if match is None:
                continue
            sentence = sentences.get(int(match.group("sentence_id")))
            if sentence is None or not str(sentence.get(transcript_key, "")).strip():
                continue
            candidates.append((audio_path, sentence))
        candidates_by_dialect[dialect] = candidates

    selected = _balanced_sample(candidates_by_dialect, catalog_size, seed)
    if not selected:
        raise ValueError("No SwissDial audio files with matching Swiss German transcripts were found.")

    output_clips = output_dir / "clips"
    output_clips.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, object]] = []
    for dialect, source_audio, sentence in selected:
        destination = output_clips / source_audio.name
        if overwrite or not destination.exists():
            shutil.copy2(source_audio, destination)

        sentence_id = int(sentence["id"])
        record: dict[str, object] = {
            "id": f"{dialect}-{sentence_id:04d}",
            "audio_path": f"clips/{destination.name}",
            "reference_transcript": str(sentence[f"ch_{dialect}"]).strip(),
            "source": "ETH SwissDial 1.1",
            "dialect": dialect.upper(),
            "dialect_name": DIALECT_NAMES[dialect],
            "sentence_id": sentence_id,
            "standard_german_transcript": str(sentence.get("de", "")).strip(),
            "topic": str(sentence.get("thema", "")).strip(),
            "language": "gsw",
        }
        if sentence.get("code_switching") is not None:
            record["code_switching"] = bool(sentence["code_switching"])
        manifest_records.append(record)

    _write_manifest(output_dir, manifest_records)
    return len(manifest_records)


def import_archive(archive_path: Path, output_dir: Path, overwrite: bool = False) -> int:
    """Extract valid clips and write a JSONL manifest for the test bench."""
    output_clips = output_dir / "clips"
    output_clips.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        tsv_name = next((name for name in archive.namelist() if name.endswith(".tsv") and "__MACOSX" not in name), None)
        if tsv_name is None:
            raise ValueError("The archive does not contain a SwissDial TSV export.")

        clip_entries = {
            Path(name).name: name
            for name in archive.namelist()
            if "/clips/" in name and name.lower().endswith((".mp3", ".wav", ".m4a")) and "__MACOSX" not in name
        }
        with archive.open(tsv_name) as tsv_file:
            rows = csv.DictReader(io.TextIOWrapper(tsv_file, encoding="utf-8"), delimiter="\t")
            records = list(rows)

        manifest_records: list[dict[str, str]] = []
        for row in records:
            if row.get("clip_is_valid", "true").lower() not in {"1", "true", "yes"}:
                continue
            clip_path = row.get("clip_path", "").strip()
            clip_id = row.get("clip_id", "").strip()
            sentence = row.get("sentence", "").strip()
            if not clip_path or not clip_id:
                continue

            archive_clip_path = clip_entries.get(Path(clip_path).name)
            if archive_clip_path is None:
                raise ValueError(f"Clip referenced by TSV is absent from archive: {clip_path}")
            destination = output_clips / Path(clip_path).name
            if destination.exists() and not overwrite:
                pass
            else:
                with archive.open(archive_clip_path) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

            manifest_records.append(
                {
                    "id": clip_id,
                    "audio_path": f"clips/{destination.name}",
                    "reference_transcript": sentence,
                    "source": "SwissDial",
                    "canton": row.get("canton", ""),
                    "zipcode": row.get("zipcode", ""),
                    "speaker_id": row.get("client_id", ""),
                    "language": "gsw",
                }
            )

    _write_manifest(output_dir, manifest_records)
    return len(manifest_records)


def main() -> None:
    arguments = parse_arguments()
    if arguments.examples_only:
        example_count = build_example_pool(arguments.source, arguments.output_dir)
        print(f"Wrote {example_count} leakage-filtered example sentences to {arguments.output_dir / 'examples.jsonl'}.")
        return
    if arguments.source.is_dir():
        count = import_official_dataset(
            arguments.source,
            arguments.output_dir,
            arguments.overwrite,
            arguments.catalog_size,
            arguments.dialects,
            arguments.seed,
        )
        example_count = build_example_pool(arguments.source, arguments.output_dir)
        print(f"Wrote {example_count} leakage-filtered example sentences for prompting.")
    else:
        count = import_archive(arguments.source, arguments.output_dir, arguments.overwrite)
    print(f"Imported {count} SwissDial clips into {arguments.output_dir}.")


if __name__ == "__main__":
    main()