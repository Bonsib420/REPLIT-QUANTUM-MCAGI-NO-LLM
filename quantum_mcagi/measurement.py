"""Measurement utilities — the quantum analogue of inference.

Measurement collapses a quantum state to a definite outcome.  Repeated
measurements build a probability distribution that the orchestrator uses
for reasoning without any language model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantum_mcagi.quantum_state import QuantumState


@dataclass(frozen=True)
class MeasurementResult:
    """Outcome of a single quantum measurement."""

    outcome: int
    probability: float


def measure(
    state: QuantumState,
    shots: int = 1,
    rng: np.random.Generator | None = None,
) -> list[MeasurementResult]:
    """Perform *shots* measurements on *state* and return results."""
    if shots < 1:
        raise ValueError("shots must be >= 1")
    rng = rng or np.random.default_rng()
    probs = state.probabilities
    outcomes = rng.choice(len(probs), size=shots, p=probs)
    unique, counts = np.unique(outcomes, return_counts=True)
    return [
        MeasurementResult(outcome=int(o), probability=float(c / shots))
        for o, c in zip(unique, counts)
    ]


def entropy(state: QuantumState) -> float:
    """Shannon entropy of the probability distribution (in bits)."""
    probs = state.probabilities
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))
