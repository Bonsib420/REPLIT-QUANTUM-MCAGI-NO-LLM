"""
quantum_state.py — Quantum state representation.

A QuantumState holds a complex amplitude vector over a discrete basis.
The state can be in superposition across all basis elements simultaneously.
Normalisation is enforced so that total probability always sums to 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class QuantumState:
    """A pure quantum state |ψ⟩ represented as a normalised complex vector.

    Parameters
    ----------
    n_basis : int
        Number of basis states (dimension of the Hilbert space).
    amplitudes : array-like, optional
        Initial complex amplitudes.  Defaults to an equal superposition of
        all basis states.
    label : str, optional
        Human-readable name for the state.
    """

    def __init__(
        self,
        n_basis: int,
        amplitudes: NDArray | None = None,
        label: str = "state",
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")
        self.n_basis = n_basis
        self.label = label
        self._rng = rng if rng is not None else np.random.default_rng()

        if amplitudes is not None:
            arr = np.asarray(amplitudes, dtype=complex)
            if arr.shape != (n_basis,):
                raise ValueError(
                    f"amplitudes shape {arr.shape} != ({n_basis},)"
                )
            self._amplitudes = arr
        else:
            # Equal superposition: |ψ⟩ = 1/√n Σ|i⟩
            self._amplitudes = np.ones(n_basis, dtype=complex) / np.sqrt(n_basis)

        self._normalise()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def amplitudes(self) -> NDArray:
        """Complex amplitude vector."""
        return self._amplitudes.copy()

    @property
    def probabilities(self) -> NDArray:
        """Born-rule probability for each basis state: |αᵢ|²."""
        return (np.abs(self._amplitudes) ** 2).real

    # ------------------------------------------------------------------
    # Quantum operations
    # ------------------------------------------------------------------

    def apply_hadamard(self) -> "QuantumState":
        """Apply a Hadamard-inspired transform to put the state into equal
        superposition regardless of its current amplitudes."""
        self._amplitudes = np.ones(self.n_basis, dtype=complex) / np.sqrt(
            self.n_basis
        )
        return self

    def apply_phase(self, index: int, phase: float) -> "QuantumState":
        """Rotate the phase of a single basis state by ``phase`` radians."""
        if not (0 <= index < self.n_basis):
            raise IndexError(f"index {index} out of range [0, {self.n_basis})")
        self._amplitudes[index] *= np.exp(1j * phase)
        self._normalise()
        return self

    def apply_rotation(self, theta: float) -> "QuantumState":
        """Apply a global real rotation by angle ``theta``.

        In 2-D this corresponds to a Bloch-sphere Y-rotation; for
        higher-dimensional spaces the rotation is applied pairwise across
        adjacent basis states.
        """
        if self.n_basis == 2:
            rot = np.array(
                [
                    [np.cos(theta / 2), -np.sin(theta / 2)],
                    [np.sin(theta / 2), np.cos(theta / 2)],
                ],
                dtype=complex,
            )
            self._amplitudes = rot @ self._amplitudes
        else:
            # Generalised: rotate each adjacent pair
            for i in range(0, self.n_basis - 1, 2):
                pair = self._amplitudes[i : i + 2]
                rot = np.array(
                    [
                        [np.cos(theta / 2), -np.sin(theta / 2)],
                        [np.sin(theta / 2), np.cos(theta / 2)],
                    ],
                    dtype=complex,
                )
                self._amplitudes[i : i + 2] = rot @ pair
        self._normalise()
        return self

    def interfere(self, other: "QuantumState", weight: float = 0.5) -> "QuantumState":
        """Quantum interference: blend amplitudes with ``other`` state.

        Parameters
        ----------
        other : QuantumState
            Must have the same ``n_basis``.
        weight : float
            Mixing weight in [0, 1].  0 keeps ``self`` unchanged, 1 replaces
            it fully with ``other``.
        """
        if other.n_basis != self.n_basis:
            raise ValueError("States must have the same n_basis to interfere")
        self._amplitudes = (
            (1 - weight) * self._amplitudes + weight * other._amplitudes
        )
        self._normalise()
        return self

    def collapse(self, outcome: int) -> "QuantumState":
        """Post-measurement collapse: project onto a single basis state."""
        if not (0 <= outcome < self.n_basis):
            raise IndexError(f"outcome {outcome} out of range")
        new_amps = np.zeros(self.n_basis, dtype=complex)
        new_amps[outcome] = 1.0
        self._amplitudes = new_amps
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise(self) -> None:
        norm = np.linalg.norm(self._amplitudes)
        if norm == 0:
            # Recover: equal superposition
            self._amplitudes = np.ones(self.n_basis, dtype=complex) / np.sqrt(
                self.n_basis
            )
        else:
            self._amplitudes /= norm

    def copy(self) -> "QuantumState":
        """Return a deep copy of this state."""
        return QuantumState(
            self.n_basis,
            amplitudes=self._amplitudes.copy(),
            label=self.label,
            rng=self._rng,
        )

    def __repr__(self) -> str:
        probs = self.probabilities
        items = ", ".join(f"|{i}⟩:{p:.3f}" for i, p in enumerate(probs))
        return f"QuantumState('{self.label}', [{items}])"
