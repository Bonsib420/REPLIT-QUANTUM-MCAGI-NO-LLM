"""Quantum Agent — the fundamental reasoning unit of the MCAGI system.

Each agent maintains its own quantum state and a wave-function of possible
actions.  Instead of prompting an LLM, the agent resolves decisions through
quantum collapse biased by a fitness function derived from its environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantum_mcagi.measurement import MeasurementResult, measure
from quantum_mcagi.quantum_state import QuantumState
from quantum_mcagi.wave_function import Decision, WaveFunction


@dataclass
class QuantumAgent:
    """A single agent that reasons via quantum state evolution."""

    name: str
    n_qubits: int = 3
    _state: QuantumState = field(init=False)
    _memory: list[MeasurementResult] = field(default_factory=list, init=False)
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, init=False)

    def __post_init__(self) -> None:
        self._state = QuantumState.uniform(self.n_qubits)

    @property
    def state(self) -> QuantumState:
        return self._state

    def perceive(self, observation: np.ndarray) -> None:
        """Integrate an external observation by rotating the internal state.

        The observation is treated as a diagonal phase-kick: each amplitude
        picks up a phase proportional to the observation value at that index.
        """
        obs = np.asarray(observation, dtype=np.float64)
        dim = self._state.dimension
        if len(obs) < dim:
            obs = np.pad(obs, (0, dim - len(obs)))
        phases = np.exp(1j * obs[:dim])
        new_amps = self._state.amplitudes * phases
        self._state = QuantumState(new_amps)

    def decide(self, options: list[str], fitness: list[float]) -> Decision:
        """Choose among *options* weighted by *fitness* via wave-function collapse."""
        wf = WaveFunction()
        for label in options:
            wf.add(label)
        return wf.resolve(fitness, rng=self._rng)

    def measure(self, shots: int = 100) -> list[MeasurementResult]:
        """Sample the internal state and cache results in memory."""
        results = measure(self._state, shots=shots, rng=self._rng)
        self._memory.extend(results)
        return results

    def act(self, actions: list[str], fitness: list[float]) -> dict[str, Any]:
        """Full perceive→decide→measure cycle. Returns a summary dict."""
        decision = self.decide(actions, fitness)
        results = self.measure(shots=50)
        return {
            "agent": self.name,
            "decision": decision.label,
            "measurement_distribution": {r.outcome: r.probability for r in results},
        }

    def __repr__(self) -> str:
        return f"QuantumAgent(name={self.name!r}, n_qubits={self.n_qubits})"
