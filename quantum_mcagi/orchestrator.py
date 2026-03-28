"""
orchestrator.py — Multi-agent quantum orchestrator.

The QuantumOrchestrator coordinates a population of QuantumAgents, enables
entanglement between selected agent pairs, and runs collective inference
episodes.  It implements a simplified Quantum Multi-Context AGI (MCAGI)
control loop:

  1. Distribute an observation to all agents.
  2. Each agent deliberates (optionally with a shared oracle).
  3. Collect individual decisions.
  4. Aggregate via quantum voting (amplitude-weighted majority).
  5. Broadcast rewards and update all agents.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .agent import QuantumAgent
from .entanglement import EntanglementRegistry
from .quantum_state import QuantumState


class QuantumOrchestrator:
    """Coordinates multiple QuantumAgents in a shared environment.

    Parameters
    ----------
    agents : list[QuantumAgent]
        The agents to orchestrate.
    shared_belief_dim : int
        Dimension of the orchestrator's own belief QuantumState.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        agents: list[QuantumAgent],
        shared_belief_dim: int = 16,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not agents:
            raise ValueError("Must provide at least one agent")
        self.agents = list(agents)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._registry = EntanglementRegistry(rng=self._rng)
        self._step = 0
        self._shared_belief = QuantumState(
            shared_belief_dim,
            label="orchestrator_belief",
            rng=self._rng,
        )
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def add_agent(self, agent: QuantumAgent) -> "QuantumOrchestrator":
        self.agents.append(agent)
        return self

    def entangle_agents(
        self,
        agent_a: QuantumAgent,
        agent_b: QuantumAgent,
        correlation: float = 0.8,
    ) -> "QuantumOrchestrator":
        """Create quantum entanglement between two agents' belief states."""
        self._registry.entangle(
            agent_a.belief, agent_b.belief, correlation=correlation
        )
        return self

    # ------------------------------------------------------------------
    # Collective inference
    # ------------------------------------------------------------------

    def run_step(
        self,
        observation: dict[str, Any],
        oracle: Callable[[str], float] | None = None,
        reward_fn: Callable[[str], float] | None = None,
        strong: bool = False,
    ) -> str:
        """Run one collective inference step.

        Parameters
        ----------
        observation : dict[str, Any]
            Current environment observation.
        oracle : callable, optional
            Shared oracle that scores actions.
        reward_fn : callable, optional
            Maps the chosen collective action to a scalar reward, which is
            broadcast to all agents.
        strong : bool
            Whether to use projective (strong) measurement for each agent.

        Returns
        -------
        str
            The collectively chosen action.
        """
        self._step += 1

        # 1. Update shared belief
        for key, val in observation.items():
            if isinstance(val, (int, float)):
                idx = hash(key) % self._shared_belief.n_basis
                self._shared_belief.apply_phase(idx, np.tanh(float(val)) * np.pi)

        # 2. Each agent deliberates and votes
        votes: dict[str, float] = {}
        agent_actions: dict[str, str] = {}

        for agent in self.agents:
            action = agent.step(observation, oracle=oracle, strong=strong)
            agent_actions[agent.name] = action
            # Weight vote by agent's confidence (top-1 probability)
            top = agent.decision_wf.top_k(1)
            confidence = top[0][1] if top else 1.0 / max(len(agent.actions), 1)
            votes[action] = votes.get(action, 0.0) + confidence

        # 3. Quantum voting: choose action proportional to summed confidence
        total = sum(votes.values())
        if total == 0:
            collective_action = self.agents[0].actions[0]
        else:
            action_list = list(votes.keys())
            probs = np.array([votes[a] / total for a in action_list])
            idx = int(self._rng.choice(len(action_list), p=probs))
            collective_action = action_list[idx]

        # 4. Compute reward and update agents
        reward = reward_fn(collective_action) if reward_fn else 0.0
        for agent in self.agents:
            agent.learn(agent_actions[agent.name], reward, observation)

        # Record
        record = {
            "step": self._step,
            "observation": observation,
            "agent_actions": agent_actions,
            "collective_action": collective_action,
            "votes": dict(votes),
            "reward": reward,
        }
        self._history.append(record)

        return collective_action

    def run_episode(
        self,
        observations: list[dict[str, Any]],
        oracle: Callable[[str], float] | None = None,
        reward_fn: Callable[[str], float] | None = None,
        strong: bool = False,
    ) -> list[str]:
        """Run multiple steps and return the list of collective actions."""
        return [
            self.run_step(obs, oracle=oracle, reward_fn=reward_fn, strong=strong)
            for obs in observations
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    @property
    def entanglement_registry(self) -> EntanglementRegistry:
        return self._registry

    def summary(self) -> str:
        """Return a human-readable summary of the last few steps."""
        lines = [f"QuantumOrchestrator — {len(self.agents)} agent(s), {self._step} steps"]
        for record in self._history[-5:]:
            agents_str = ", ".join(
                f"{n}→{a}" for n, a in record["agent_actions"].items()
            )
            lines.append(
                f"  step {record['step']:3d}: [{agents_str}] "
                f"→ collective={record['collective_action']!r}, "
                f"reward={record['reward']:.3f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        names = ", ".join(a.name for a in self.agents)
        return f"QuantumOrchestrator([{names}], step={self._step})"
