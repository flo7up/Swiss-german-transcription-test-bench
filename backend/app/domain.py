"""Core benchmark data structures independent of transport and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    kind: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    description: str | None = None


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    label: str
    deployment: str
    description: str
    capabilities: tuple[str, ...]
    transport: str = "foundry-responses"
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DatasetItem:
    id: str
    audio_path: Path
    reference_transcript: str | None
    source: str
    metadata: dict[str, str]
