"""
entanglement.py — Quantum entanglement between QuantumState objects.

Two entangled states become correlated: measuring (collapsing) one
instantly constrains the probabilities of the other, mimicking the
non-local correlations described by Bell states in quantum mechanics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .quantum_state import QuantumState


class EntangledPair:
    """A Bell-state-inspired entangled pair of QuantumStates.

    The joint state is stored as a 2D tensor (n_a × n_b), and individual
    state probabilities are obtained via the partial trace.

    Parameters
    ----------
    state_a, state_b : QuantumState
        The two states to entangle.  They may have different dimensions.
    correlation : float
        Strength of entanglement in [0, 1].  0 = independent, 1 = maximally
        entangled.
    """

    def __init__(
        self,
        state_a: QuantumState,
        state_b: QuantumState,
        correlation: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not (0.0 <= correlation <= 1.0):
            raise ValueError("correlation must be in [0, 1]")
        self.state_a = state_a
        self.state_b = state_b
        self.correlation = correlation
        self._rng = rng if rng is not None else np.random.default_rng()
        self._joint = self._build_joint()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_joint(self) -> NDArray:
        """Build a joint amplitude tensor from the product state, mixed with
        a Bell-like maximally entangled component."""
        na, nb = self.state_a.n_basis, self.state_b.n_basis
        # Product state
        product = np.outer(self.state_a.amplitudes, self.state_b.amplitudes)

        # Maximally entangled component: uniform superposition of diagonal
        n_diag = min(na, nb)
        entangled = np.zeros((na, nb), dtype=complex)
        for k in range(n_diag):
            entangled[k, k] = 1.0 / np.sqrt(n_diag)

        joint = (1.0 - self.correlation) * product + self.correlation * entangled
        norm = np.linalg.norm(joint)
        return joint / norm if norm > 0 else joint

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_a(self) -> int:
        """Measure subsystem A.  Returns outcome index and updates B's state
        based on the conditional probability P(B | A=outcome)."""
        probs_a = (np.abs(self._joint) ** 2).sum(axis=1).real
        probs_a /= probs_a.sum()
        outcome_a = int(self._rng.choice(self.state_a.n_basis, p=probs_a))
        self.state_a.collapse(outcome_a)

        # Update B: conditional distribution given A's outcome
        cond_b = np.abs(self._joint[outcome_a, :]) ** 2
        norm_b = cond_b.sum()
        if norm_b > 0:
            cond_b /= norm_b
        else:
            cond_b = np.ones(self.state_b.n_basis) / self.state_b.n_basis

        new_amps_b = np.sqrt(cond_b).astype(complex)
        self.state_b._amplitudes = new_amps_b
        self.state_b._normalise()
        return outcome_a

    def measure_b(self) -> int:
        """Measure subsystem B.  Returns outcome index and updates A."""
        probs_b = (np.abs(self._joint) ** 2).sum(axis=0).real
        probs_b /= probs_b.sum()
        outcome_b = int(self._rng.choice(self.state_b.n_basis, p=probs_b))
        self.state_b.collapse(outcome_b)

        cond_a = np.abs(self._joint[:, outcome_b]) ** 2
        norm_a = cond_a.sum()
        if norm_a > 0:
            cond_a /= norm_a
        else:
            cond_a = np.ones(self.state_a.n_basis) / self.state_a.n_basis

        new_amps_a = np.sqrt(cond_a).astype(complex)
        self.state_a._amplitudes = new_amps_a
        self.state_a._normalise()
        return outcome_b

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def reset(self) -> "EntangledPair":
        """Rebuild the joint state from current (possibly updated) amplitudes."""
        self._joint = self._build_joint()
        return self

    def __repr__(self) -> str:
        return (
            f"EntangledPair({self.state_a.label!r} ↔ {self.state_b.label!r}, "
            f"correlation={self.correlation:.2f})"
        )


class EntanglementRegistry:
    """Registry that tracks all active entangled pairs in the system.

    Parameters
    ----------
    rng : np.random.Generator, optional
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._pairs: list[EntangledPair] = []
        self._rng = rng if rng is not None else np.random.default_rng()

    def entangle(
        self,
        state_a: QuantumState,
        state_b: QuantumState,
        correlation: float = 1.0,
    ) -> EntangledPair:
        """Create and register an entangled pair."""
        pair = EntangledPair(state_a, state_b, correlation=correlation, rng=self._rng)
        self._pairs.append(pair)
        return pair

    def disentangle(self, pair: EntangledPair) -> None:
        """Remove a pair from the registry."""
        self._pairs = [p for p in self._pairs if p is not pair]

    @property
    def pairs(self) -> list[EntangledPair]:
        return list(self._pairs)

    def __repr__(self) -> str:
        return f"EntanglementRegistry({len(self._pairs)} active pairs)"
