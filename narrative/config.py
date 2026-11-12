"""Settings of the narrative extension that the magma configuration does not have."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NarrativeConfig:
    narrator: str = "narrator"  # speaker name of narration turns: facts of the world, with no claim and no actor


DEFAULT_NARRATIVE = NarrativeConfig()
