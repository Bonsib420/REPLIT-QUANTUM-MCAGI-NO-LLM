"""Entanglement channel for correlating multiple agents.

When two agents share an entanglement channel their internal quantum states
become correlated — measuring one instantly biases the other, enabling
coordinated multi-agent behaviour without explicit message passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from quantum_mcagi.quantum_state import QuantumState

if TYPE_CHECKING:
    from quantum_mcagi.agent import QuantumAgent


@dataclass
class EntanglementChannel:
    """A shared quantum channel linking two or more agents."""

    agents: list[QuantumAgent] = field(default_factory=list)
    _correlation: float = 0.0

    def link(self, *agents: QuantumAgent) -> None:
        """Add agents to the entanglement channel."""
        for a in agents:
            if a not in self.agents:
                self.agents.append(a)
        self._update_correlation()

    def _update_correlation(self) -> None:
        if len(self.agents) < 2:
            self._correlation = 0.0
            return
        vecs = [a.state.amplitudes for a in self.agents]
        overlaps: list[float] = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                min_len = min(len(vecs[i]), len(vecs[j]))
                overlap = float(
                    np.abs(np.vdot(vecs[i][:min_len], vecs[j][:min_len]))
                )
                overlaps.append(overlap)
        self._correlation = float(np.mean(overlaps)) if overlaps else 0.0

    @property
    def correlation(self) -> float:
        """Return the average pairwise overlap (0 = uncorrelated, 1 = maximally entangled)."""
        self._update_correlation()
        return self._correlation

    def synchronise(self) -> None:
        """Push each agent's state toward the channel mean, increasing correlation."""
        if len(self.agents) < 2:
            return
        max_dim = max(a.state.dimension for a in self.agents)
        padded = []
        for a in self.agents:
            v = a.state.amplitudes
            p = np.zeros(max_dim, dtype=np.complex128)
            p[: len(v)] = v
            padded.append(p)
        mean_vec = np.mean(padded, axis=0)
        alpha = 0.3
        for i, a in enumerate(self.agents):
            blended = (1 - alpha) * padded[i] + alpha * mean_vec
            dim = a.state.dimension
            a._state = QuantumState(blended[:dim])
