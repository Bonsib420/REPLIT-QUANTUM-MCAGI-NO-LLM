"""Quantum state representation using amplitude vectors.

A QuantumState is a normalised complex amplitude vector of size 2**n_qubits.
It supports superposition, amplitude manipulation, and probabilistic collapse.
"""

from __future__ import annotations

import numpy as np


class QuantumState:
    """Immutable quantum state represented by a normalised amplitude vector."""

    def __init__(self, amplitudes: np.ndarray | list[complex]) -> None:
        vec = np.asarray(amplitudes, dtype=np.complex128)
        if vec.ndim != 1 or len(vec) == 0:
            raise ValueError("amplitudes must be a non-empty 1-D array")
        n = len(vec)
        if n & (n - 1) != 0:
            raise ValueError(f"length must be a power of 2, got {n}")
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("zero vector cannot represent a quantum state")
        self._amplitudes: np.ndarray = vec / norm

    @classmethod
    def basis(cls, index: int, n_qubits: int) -> QuantumState:
        """Create a computational basis state |index⟩ for *n_qubits* qubits."""
        dim = 1 << n_qubits
        if not 0 <= index < dim:
            raise ValueError(f"index {index} out of range for {n_qubits} qubits")
        amps = np.zeros(dim, dtype=np.complex128)
        amps[index] = 1.0
        return cls(amps)

    @classmethod
    def uniform(cls, n_qubits: int) -> QuantumState:
        """Create an equal superposition over all basis states."""
        dim = 1 << n_qubits
        return cls(np.ones(dim, dtype=np.complex128))

    @property
    def amplitudes(self) -> np.ndarray:
        return self._amplitudes.copy()

    @property
    def probabilities(self) -> np.ndarray:
        return np.abs(self._amplitudes) ** 2

    @property
    def n_qubits(self) -> int:
        return int(np.log2(len(self._amplitudes)))

    @property
    def dimension(self) -> int:
        return len(self._amplitudes)

    def apply(self, unitary: np.ndarray) -> QuantumState:
        """Return a new state produced by applying a unitary matrix."""
        u = np.asarray(unitary, dtype=np.complex128)
        if u.shape != (self.dimension, self.dimension):
            raise ValueError("unitary dimensions do not match state dimension")
        return QuantumState(u @ self._amplitudes)

    def collapse(self, rng: np.random.Generator | None = None) -> int:
        """Collapse the state and return the measured basis-state index."""
        rng = rng or np.random.default_rng()
        probs = self.probabilities
        return int(rng.choice(len(probs), p=probs))

    def __repr__(self) -> str:
        return f"QuantumState(n_qubits={self.n_qubits}, dim={self.dimension})"
