"""Wave-function utilities for decision-making through superposition and collapse.

Instead of an LLM generating text, decisions are encoded as quantum amplitudes
and resolved by probabilistic collapse — biased toward high-fitness options.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantum_mcagi.quantum_state import QuantumState


@dataclass(frozen=True)
class Decision:
    """One possible outcome with a human-readable label."""

    label: str
    index: int


@dataclass
class WaveFunction:
    """A superposition of possible decisions weighted by fitness scores."""

    decisions: list[Decision] = field(default_factory=list)

    def add(self, label: str) -> Decision:
        """Append a new decision option and return it."""
        d = Decision(label=label, index=len(self.decisions))
        self.decisions.append(d)
        return d

    def resolve(
        self,
        fitness: np.ndarray | list[float],
        rng: np.random.Generator | None = None,
    ) -> Decision:
        """Collapse the wave function using *fitness* as amplitude weights.

        Higher fitness → higher probability, but lower-fitness options still
        have a non-zero chance (quantum-style exploration).
        """
        if len(fitness) != len(self.decisions):
            raise ValueError("fitness length must match number of decisions")
        fit = np.asarray(fitness, dtype=np.float64)
        if np.any(fit < 0):
            raise ValueError("fitness values must be non-negative")
        if np.all(fit == 0):
            fit = np.ones_like(fit)
        n_qubits = int(np.ceil(np.log2(max(len(fit), 2))))
        dim = 1 << n_qubits
        amplitudes = np.zeros(dim, dtype=np.complex128)
        amplitudes[: len(fit)] = np.sqrt(fit)
        state = QuantumState(amplitudes)
        idx = state.collapse(rng=rng)
        if idx >= len(self.decisions):
            idx = int(rng.choice(len(self.decisions))) if rng else 0
        return self.decisions[idx]
