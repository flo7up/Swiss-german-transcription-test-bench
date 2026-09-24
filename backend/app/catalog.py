"""Load the editable model registry and local SwissDial manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .domain import DatasetItem, ModelDefinition, ParameterSpec


def load_models(path: Path) -> list[ModelDefinition]:
    """Load configured Foundry deployments from a JSON array."""
    raw_models = json.loads(path.read_text(encoding="utf-8"))
    return [
        ModelDefinition(
            id=model["id"],
            label=model["label"],
            deployment=model["deployment"],
            description=model["description"],
            capabilities=tuple(model.get("capabilities", [])),
            transport=model.get("transport", "foundry-responses"),
            parameters=tuple(
                ParameterSpec(
                    name=parameter["name"],
                    label=parameter["label"],
                    kind=parameter["kind"],
                    default=parameter["default"],
                    minimum=parameter.get("minimum"),
                    maximum=parameter.get("maximum"),
                    step=parameter.get("step"),
                    description=parameter.get("description"),
                )
                for parameter in model.get("parameters", [])
            ),
        )
        for model in raw_models
    ]


def load_voice_options(path: Path) -> list[dict[str, Any]]:
    """Load selectable voice model and managed API metadata for the UI."""
    if not path.exists():
        return []
    options = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(options, list):
        raise ValueError(f"Voice options must be a JSON array: {path}")
    return options


def load_dialect_atlas(path: Path) -> dict[str, Any]:
    """Load dialect profiles and derive speaker estimates from canton demographics."""
    if not path.exists():
        return {"dialects": [], "cantons": {}, "sources": []}
    atlas = json.loads(path.read_text(encoding="utf-8"))
    cantons: dict[str, dict[str, Any]] = atlas.get("cantons", {})

    def german_speakers(code: str) -> float:
        canton = cantons.get(code)
        return canton["population"] * canton["german_share"] if canton else 0.0

    total_population = sum(canton["population"] for canton in cantons.values())
    total_german = sum(german_speakers(code) for code in cantons)
    dialects: list[dict[str, Any]] = []
    for dialect in atlas.get("dialects", []):
        region_speakers = sum(german_speakers(code) for code in dialect.get("region_cantons", []))
        core_speakers = german_speakers(dialect["code"])
        dialects.append(
            {
                **dialect,
                "core_speakers": round(core_speakers),
                "region_speakers": round(region_speakers),
                "share_of_german_speakers": region_speakers / total_german if total_german else None,
                "share_of_population": region_speakers / total_population if total_population else None,
            }
        )
    dialects.sort(key=lambda entry: entry["region_speakers"], reverse=True)
    covered = sum(entry["region_speakers"] for entry in dialects)
    return {
        **atlas,
        "dialects": dialects,
        "total_population": total_population,
        "total_german_speakers": round(total_german),
        "covered_german_speakers": covered,
        "coverage_share": covered / total_german if total_german else None,
    }


def load_dataset_items(manifest_path: Path) -> list[DatasetItem]:
    """Load JSONL records, resolving audio paths relative to the manifest file."""
    if not manifest_path.exists():
        return []

    items: list[DatasetItem] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record: dict[str, Any] = json.loads(line)
            item_id = str(record["id"])
            audio_path = manifest_path.parent / str(record["audio_path"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid dataset record on line {line_number} of {manifest_path}") from error

        metadata = {
            key: str(value)
            for key, value in record.items()
            if key not in {"id", "audio_path", "reference_transcript", "source"} and value not in (None, "")
        }
        items.append(
            DatasetItem(
                id=item_id,
                audio_path=audio_path,
                reference_transcript=record.get("reference_transcript"),
                source=str(record.get("source", "Unknown")),
                metadata=metadata,
            )
        )
    return items