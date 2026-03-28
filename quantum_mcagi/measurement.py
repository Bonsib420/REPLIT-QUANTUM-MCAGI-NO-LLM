"""
measurement.py — Quantum measurement operations.

Measurement collapses a WaveFunction or QuantumState to a definite outcome
according to Born-rule probabilities.  Multiple measurement strategies are
provided: projective, weak (partial collapse), and repeated (averaging).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .quantum_state import QuantumState
from .wave_function import WaveFunction


class Measurement:
    """Collection of measurement strategies.

    Parameters
    ----------
    rng : np.random.Generator, optional
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Projective measurement
    # ------------------------------------------------------------------

    def projective(self, state: QuantumState) -> tuple[int, float]:
        """Strong (projective) measurement.  Collapses ``state`` to one basis
        element and returns ``(outcome_index, probability)``.

        After this call ``state`` is in a definite eigenstate.
        """
        probs = state.probabilities
        outcome = int(self._rng.choice(state.n_basis, p=probs))
        prob = float(probs[outcome])
        state.collapse(outcome)
        return outcome, prob

    def projective_wf(self, wf: WaveFunction) -> tuple[str, float]:
        """Projective measurement on a WaveFunction.

        Returns ``(option_name, probability)``.
        """
        outcome_idx, prob = self.projective(wf.state)
        return wf.options[outcome_idx], prob

    # ------------------------------------------------------------------
    # Weak measurement
    # ------------------------------------------------------------------

    def weak(
        self, state: QuantumState, strength: float = 0.3
    ) -> tuple[int, float]:
        """Weak (partial-collapse) measurement.

        The state is partially collapsed toward the outcome: a ``strength``
        in (0, 1) blends the post-measurement eigenstate with the prior state.

        Returns ``(outcome_index, probability)``.
        """
        if not (0 < strength <= 1):
            raise ValueError("strength must be in (0, 1]")
        probs = state.probabilities
        outcome = int(self._rng.choice(state.n_basis, p=probs))
        prob = float(probs[outcome])

        # Build post-measurement eigenstate
        post = np.zeros(state.n_basis, dtype=complex)
        post[outcome] = 1.0

        # Partial collapse
        new_amps = (1 - strength) * state._amplitudes + strength * post
        state._amplitudes = new_amps
        state._normalise()
        return outcome, prob

    def weak_wf(
        self, wf: WaveFunction, strength: float = 0.3
    ) -> tuple[str, float]:
        """Weak measurement on a WaveFunction."""
        outcome_idx, prob = self.weak(wf.state, strength=strength)
        return wf.options[outcome_idx], prob

    # ------------------------------------------------------------------
    # Repeated measurement (expectation estimation)
    # ------------------------------------------------------------------

    def repeated(
        self, state: QuantumState, n_shots: int = 1024
    ) -> NDArray:
        """Estimate outcome probabilities via repeated sampling (without
        collapsing the state).

        Returns a probability array of length ``state.n_basis``.
        """
        probs = state.probabilities
        counts = np.zeros(state.n_basis, dtype=int)
        outcomes = self._rng.choice(state.n_basis, size=n_shots, p=probs)
        for o in outcomes:
            counts[o] += 1
        return counts / n_shots

    def expectation(
        self, state: QuantumState, observable: NDArray, n_shots: int = 1024
    ) -> float:
        """Estimate ⟨ψ|O|ψ⟩ for a diagonal observable (1-D array of
        eigenvalues) using repeated sampling.

        Parameters
        ----------
        state : QuantumState
        observable : array-like, shape (n_basis,)
            Eigenvalues of the observable.
        n_shots : int
        """
        obs = np.asarray(observable, dtype=float)
        if obs.shape != (state.n_basis,):
            raise ValueError("observable must have shape (n_basis,)")
        freq = self.repeated(state, n_shots=n_shots)
        return float(np.dot(freq, obs))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return "Measurement()"
