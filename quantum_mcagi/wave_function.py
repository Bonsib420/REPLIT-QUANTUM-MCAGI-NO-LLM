"""
wave_function.py — Wave function over a continuous or discrete decision space.

A WaveFunction wraps a QuantumState and associates each basis index with a
named option (action, belief, hypothesis, etc.).  It exposes the quantum
probability distribution used for sampling and amplitude-weighted scoring.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .quantum_state import QuantumState


class WaveFunction:
    """Maps a QuantumState onto a named option space.

    Parameters
    ----------
    options : list[str]
        The discrete options (actions, hypotheses, …) that form the basis.
    label : str, optional
        Human-readable name.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        options: list[str],
        label: str = "wave_function",
        rng: np.random.Generator | None = None,
    ) -> None:
        if not options:
            raise ValueError("options list must not be empty")
        self.options = list(options)
        self.label = label
        self._rng = rng if rng is not None else np.random.default_rng()
        self._state = QuantumState(
            len(options), label=label, rng=self._rng
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> QuantumState:
        return self._state

    @property
    def probabilities(self) -> dict[str, float]:
        """Return {option: probability} mapping."""
        return dict(zip(self.options, self._state.probabilities.tolist()))

    @property
    def amplitudes(self) -> NDArray:
        return self._state.amplitudes

    # ------------------------------------------------------------------
    # Amplitude manipulation
    # ------------------------------------------------------------------

    def amplify(self, option: str, factor: float = 1.5) -> "WaveFunction":
        """Increase the amplitude of ``option`` by ``factor`` (> 1 boosts it,
        < 1 suppresses it).  The state is renormalised afterward."""
        idx = self._index(option)
        amps = self._state._amplitudes.copy()
        amps[idx] *= factor
        self._state._amplitudes = amps
        self._state._normalise()
        return self

    def suppress(self, option: str, factor: float = 0.5) -> "WaveFunction":
        """Suppress the amplitude of ``option`` (shorthand for amplify < 1)."""
        return self.amplify(option, factor)

    def interfere_with(
        self, other: "WaveFunction", weight: float = 0.5
    ) -> "WaveFunction":
        """Quantum interference: blend this wave function with ``other``."""
        if other.options != self.options:
            raise ValueError("WaveFunctions must share the same option space")
        self._state.interfere(other._state, weight=weight)
        return self

    def apply_oracle(self, oracle_fn: Callable[[str], float]) -> "WaveFunction":
        """Phase-kick each option proportionally to ``oracle_fn(option)``.

        The oracle assigns a real score in [-π, π] to each option.  Higher
        scores increase constructive interference for that option (Grover-
        inspired amplitude amplification).
        """
        for i, opt in enumerate(self.options):
            score = float(oracle_fn(opt))
            self._state.apply_phase(i, score)
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self) -> str:
        """Draw one option proportional to Born-rule probabilities."""
        probs = self._state.probabilities
        idx = int(self._rng.choice(len(self.options), p=probs))
        return self.options[idx]

    def top_k(self, k: int = 1) -> list[tuple[str, float]]:
        """Return the ``k`` options with highest probability."""
        probs = self._state.probabilities
        indices = np.argsort(probs)[::-1][:k]
        return [(self.options[i], float(probs[i])) for i in indices]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _index(self, option: str) -> int:
        try:
            return self.options.index(option)
        except ValueError:
            raise KeyError(f"Option '{option}' not in WaveFunction '{self.label}'")

    def reset(self) -> "WaveFunction":
        """Reset to equal superposition over all options."""
        self._state.apply_hadamard()
        return self

    def copy(self) -> "WaveFunction":
        wf = WaveFunction(list(self.options), label=self.label, rng=self._rng)
        wf._state = self._state.copy()
        return wf

    def __repr__(self) -> str:
        top = self.top_k(min(3, len(self.options)))
        items = ", ".join(f"{o}:{p:.3f}" for o, p in top)
        return f"WaveFunction('{self.label}', top=[{items}])"
