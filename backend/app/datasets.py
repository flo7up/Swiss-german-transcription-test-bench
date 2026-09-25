"""Validate a local audio dataset ZIP before making it visible to benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

from .catalog import load_dataset_items


MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 5_000
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm"}


def _safe_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid archive path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute() or path.as_posix() != value or len(value) > 240
        or any(part in {".", ".."} or len(part) > 120 for part in value.split("/"))
    ):
        raise ValueError(f"Invalid archive path: {value!r}")
    if ":" in path.parts[0]:
        raise ValueError(f"Invalid archive path: {value!r}")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{number}" for number in range(1, 10)),
                *(f"LPT{number}" for number in range(1, 10))}
    if any(
        any(character in '<>:"|?*' or ord(character) < 32 for character in part)
        or part.endswith((" ", "."))
        or part.split(".")[0].upper() in reserved
        for part in path.parts
    ):
        raise ValueError(f"Invalid archive path: {value!r}")
    return path


def _manifest_records(content: bytes, name: str, existing_ids: set[str]) -> tuple[list[dict], set[str]]:
    try:
        lines = content.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("manifest.jsonl must be UTF-8 text.") from error

    records: list[dict] = []
    audio_paths: set[str] = set()
    seen = set(existing_ids)
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on manifest line {line_number}.") from error
        if not isinstance(record, dict):
            raise ValueError(f"Manifest line {line_number} must be a JSON object.")
        item_id = record.get("id")
        if not isinstance(item_id, str) or not item_id or len(item_id) > 120 or not all(
            character.isascii() and (character.isalnum() or character in "._-") for character in item_id
        ):
            raise ValueError(f"Manifest line {line_number} needs an alphanumeric ID (letters, digits, '.', '_' or '-').")
        if item_id in seen:
            raise ValueError(f"Duplicate dataset item ID: {item_id}")
        seen.add(item_id)

        audio_path = _safe_path(record.get("audio_path"))
        if audio_path.parts[0] != "clips" or len(audio_path.parts) < 2 or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Manifest line {line_number} must reference a supported audio file under clips/.")
        for field in ("reference_transcript", "standard_german_transcript", "source"):
            if field in record and record[field] is not None and not isinstance(record[field], str):
                raise ValueError(f"Manifest line {line_number} has a non-text {field}.")
        record["audio_path"] = audio_path.as_posix()
        record["source"] = record.get("source") or name
        record["dataset"] = name
        audio_paths.add(audio_path.as_posix())
        records.append(record)
        if len(records) > MAX_ARCHIVE_ENTRIES - 1:
            raise ValueError(f"Manifest contains more than {MAX_ARCHIVE_ENTRIES - 1} items.")
    if not records:
        raise ValueError("manifest.jsonl must contain at least one audio item.")
    return records, audio_paths


def extract_dataset_archive(archive_path: Path, destination: Path, name: str, existing_ids: set[str]) -> int:
    """Extract only declared audio into a staging directory; caller publishes it atomically."""
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_ENTRIES:
            raise ValueError(f"Archive contains more than {MAX_ARCHIVE_ENTRIES} entries.")
        files: dict[str, zipfile.ZipInfo] = {}
        folded: set[str] = set()
        total_size = 0
        for info in infos:
            path = _safe_path(info.filename.rstrip("/") if info.is_dir() else info.filename)
            if info.flag_bits & 1:
                raise ValueError("Encrypted ZIP entries are not supported.")
            if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                raise ValueError("ZIP symlinks are not supported.")
            if info.is_dir():
                continue
            normalized = path.as_posix()
            if normalized.casefold() in folded:
                raise ValueError(f"Duplicate archive path: {normalized}")
            folded.add(normalized.casefold())
            files[normalized] = info
            total_size += info.file_size
            if total_size > MAX_EXTRACTED_BYTES:
                raise ValueError("Extracted dataset exceeds the 2 GiB limit.")
        manifest = files.get("manifest.jsonl")
        if manifest is None:
            raise ValueError("ZIP must contain manifest.jsonl at its root.")
        if manifest.file_size > MAX_MANIFEST_BYTES:
            raise ValueError("manifest.jsonl exceeds the 8 MiB limit.")
        with archive.open(manifest) as source:
            content = source.read(MAX_MANIFEST_BYTES + 1)
        if len(content) > MAX_MANIFEST_BYTES:
            raise ValueError("manifest.jsonl exceeds the 8 MiB limit.")
        records, audio_paths = _manifest_records(content, name, existing_ids)
        if set(files) != {"manifest.jsonl", *audio_paths}:
            missing = sorted(audio_paths - files.keys())
            unexpected = sorted(files.keys() - audio_paths - {"manifest.jsonl"})
            raise ValueError(f"ZIP audio files must match the manifest (missing: {missing}; unreferenced: {unexpected}).")
        if any(
            parent.as_posix() in audio_paths
            for path in audio_paths
            for parent in PurePosixPath(path).parents
        ):
            raise ValueError("An audio file cannot also be a directory for another clip.")

        manifest_content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")
        if len(manifest_content) > MAX_EXTRACTED_BYTES:
            raise ValueError("Extracted dataset exceeds the 2 GiB limit.")
        destination.mkdir()
        remaining = MAX_EXTRACTED_BYTES - len(manifest_content)
        for audio_path in sorted(audio_paths):
            info = files[audio_path]
            target = destination.joinpath(*PurePosixPath(audio_path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                while chunk := source.read(min(1024 * 1024, remaining + 1)):
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise ValueError("Extracted dataset exceeds the 2 GiB limit.")
                    output.write(chunk)
            if target.stat().st_size == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
        (destination / "manifest.jsonl").write_bytes(manifest_content)
        load_dataset_items(destination / "manifest.jsonl")
        return len(records)
