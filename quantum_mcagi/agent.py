"""
agent.py — Quantum-inspired AI agent.

A QuantumAgent maintains an internal quantum belief state and a set of
wave functions representing its decision spaces.  It reasons without any
LLM: all inference is performed via quantum amplitude operations,
Grover-style oracle amplification, and Born-rule sampling.

Core loop
---------
1. **Perceive**: encode observations into the belief QuantumState.
2. **Deliberate**: apply oracle functions to amplify promising actions.
3. **Decide**: sample an action from the collapsed WaveFunction.
4. **Learn**: update amplitude weights based on reward feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .quantum_state import QuantumState
from .wave_function import WaveFunction
from .measurement import Measurement


@dataclass
class Experience:
    """A single agent experience (state, action, reward)."""

    observation: dict[str, Any]
    action: str
    reward: float
    step: int


class QuantumAgent:
    """Quantum-inspired AI agent.

    Parameters
    ----------
    name : str
    actions : list[str]
        The set of actions the agent can take.
    belief_dim : int
        Dimensionality of the internal belief QuantumState.
    learning_rate : float
        Controls how strongly rewards update amplitude weights.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        name: str,
        actions: list[str],
        belief_dim: int = 8,
        learning_rate: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not actions:
            raise ValueError("actions must not be empty")
        self.name = name
        self.actions = list(actions)
        self.learning_rate = learning_rate
        self._rng = rng if rng is not None else np.random.default_rng()
        self._step = 0

        # Internal belief state
        self._belief = QuantumState(
            belief_dim, label=f"{name}_belief", rng=self._rng
        )
        # Decision wave function over actions
        self._decision_wf = WaveFunction(
            actions, label=f"{name}_decision", rng=self._rng
        )
        self._measurement = Measurement(rng=self._rng)
        self._memory: list[Experience] = []
        # Amplitude bias per action, initialised to 1.0
        self._action_bias: dict[str, float] = {a: 1.0 for a in actions}

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def perceive(self, observation: dict[str, Any]) -> "QuantumAgent":
        """Encode an observation into the belief state via phase kicks.

        Each numeric value in the observation is mapped to a phase rotation
        applied to one belief basis state (cycling through them).
        """
        numeric_values = [
            float(v) for v in observation.values() if isinstance(v, (int, float))
        ]
        for i, val in enumerate(numeric_values):
            idx = i % self._belief.n_basis
            # Normalise to [-π, π]
            phase = np.tanh(val) * np.pi
            self._belief.apply_phase(idx, phase)
        return self

    def deliberate(
        self,
        oracle: Callable[[str], float] | None = None,
    ) -> "QuantumAgent":
        """Amplify promising actions using the oracle and learned biases.

        If no oracle is provided, only the learned amplitude biases are used.
        """
        # Reset decision WF to equal superposition
        self._decision_wf.reset()

        # Apply learned biases first
        for action, bias in self._action_bias.items():
            self._decision_wf.amplify(action, factor=max(bias, 1e-6))

        # Apply external oracle if provided (Grover-style phase kick)
        if oracle is not None:
            self._decision_wf.apply_oracle(oracle)

        # Interfere decision WF with belief state projection
        belief_proj = WaveFunction(
            self.actions, label="belief_proj", rng=self._rng
        )
        # Map belief probs to action space (circular mapping)
        n = len(self.actions)
        belief_probs = self._belief.probabilities
        mapped = np.array(
            [belief_probs[i % self._belief.n_basis] for i in range(n)]
        )
        mapped_norm = mapped / mapped.sum() if mapped.sum() > 0 else np.ones(n) / n
        new_amps = np.sqrt(mapped_norm).astype(complex)
        belief_proj._state._amplitudes = new_amps
        belief_proj._state._normalise()

        self._decision_wf.interfere_with(belief_proj, weight=0.3)
        return self

    def decide(self, strong: bool = False) -> str:
        """Sample an action from the decision wave function.

        Parameters
        ----------
        strong : bool
            If True, use projective (strong) measurement, collapsing the WF.
            If False (default), use weak measurement, preserving superposition.
        """
        if strong:
            action, _ = self._measurement.projective_wf(self._decision_wf)
        else:
            action, _ = self._measurement.weak_wf(
                self._decision_wf, strength=0.4
            )
        return action

    def learn(self, action: str, reward: float, observation: dict[str, Any]) -> "QuantumAgent":
        """Update amplitude biases based on received reward.

        Positive reward increases the bias for the chosen action.
        Negative reward decreases it, allowing other actions to dominate.
        """
        self._step += 1
        self._memory.append(
            Experience(
                observation=observation,
                action=action,
                reward=reward,
                step=self._step,
            )
        )

        # Gradient-free amplitude update
        delta = self.learning_rate * reward
        old_bias = self._action_bias[action]
        new_bias = max(old_bias * np.exp(delta), 1e-4)
        self._action_bias[action] = new_bias

        # Normalise biases so max = 2.0 (prevents unbounded growth)
        max_bias = max(self._action_bias.values())
        if max_bias > 2.0:
            factor = 2.0 / max_bias
            self._action_bias = {k: v * factor for k, v in self._action_bias.items()}

        return self

    def step(
        self,
        observation: dict[str, Any],
        oracle: Callable[[str], float] | None = None,
        reward: float | None = None,
        strong: bool = False,
    ) -> str:
        """Convenience: perceive → deliberate → decide (→ learn if reward given).

        Returns the chosen action string.
        """
        self.perceive(observation)
        self.deliberate(oracle=oracle)
        action = self.decide(strong=strong)
        if reward is not None:
            self.learn(action, reward, observation)
        return action

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def belief(self) -> QuantumState:
        return self._belief

    @property
    def decision_wf(self) -> WaveFunction:
        return self._decision_wf

    @property
    def memory(self) -> list[Experience]:
        return list(self._memory)

    @property
    def action_probabilities(self) -> dict[str, float]:
        return self._decision_wf.probabilities

    def reset(self) -> "QuantumAgent":
        """Reset belief and decision WF to equal superposition."""
        self._belief.apply_hadamard()
        self._decision_wf.reset()
        return self

    def __repr__(self) -> str:
        top = self._decision_wf.top_k(min(3, len(self.actions)))
        items = ", ".join(f"{a}:{p:.3f}" for a, p in top)
        return f"QuantumAgent('{self.name}', top_actions=[{items}], step={self._step})"
