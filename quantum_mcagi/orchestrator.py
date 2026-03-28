"""Orchestrator — coordinates multiple QuantumAgents into a collective intelligence.

The orchestrator manages agent creation, entanglement, task distribution,
and consensus.  It is the top-level entry point for running the MCAGI system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantum_mcagi.agent import QuantumAgent
from quantum_mcagi.entanglement import EntanglementChannel
from quantum_mcagi.measurement import entropy


@dataclass
class Orchestrator:
    """Multi-agent orchestrator that drives collective quantum reasoning."""

    agents: list[QuantumAgent] = field(default_factory=list)
    channels: list[EntanglementChannel] = field(default_factory=list)

    def add_agent(self, name: str, n_qubits: int = 3) -> QuantumAgent:
        """Create and register a new agent."""
        agent = QuantumAgent(name=name, n_qubits=n_qubits)
        self.agents.append(agent)
        return agent

    def entangle(self, *agents: QuantumAgent) -> EntanglementChannel:
        """Create a shared entanglement channel among the given agents."""
        ch = EntanglementChannel()
        ch.link(*agents)
        self.channels.append(ch)
        return ch

    def broadcast_observation(self, observation: np.ndarray) -> None:
        """Send the same observation to every agent."""
        for agent in self.agents:
            agent.perceive(observation)

    def collective_decide(
        self,
        options: list[str],
        fitness: list[float],
    ) -> dict[str, Any]:
        """Have all agents vote and return a consensus decision.

        Each agent independently collapses its wave function.  The option
        chosen by the most agents wins (majority vote).
        """
        votes: dict[str, int] = {}
        details: list[dict[str, Any]] = []
        for agent in self.agents:
            result = agent.act(options, fitness)
            details.append(result)
            label = result["decision"]
            votes[label] = votes.get(label, 0) + 1
        winner = max(votes, key=lambda k: votes[k])
        return {
            "consensus": winner,
            "votes": votes,
            "agent_details": details,
        }

    def synchronise(self) -> None:
        """Push all entangled agents toward their channel means."""
        for ch in self.channels:
            ch.synchronise()

    def report(self) -> dict[str, Any]:
        """Return a summary of the current system state."""
        agent_info = []
        for a in self.agents:
            agent_info.append(
                {
                    "name": a.name,
                    "n_qubits": a.n_qubits,
                    "entropy": entropy(a.state),
                }
            )
        channel_info = [
            {
                "agents": [a.name for a in ch.agents],
                "correlation": ch.correlation,
            }
            for ch in self.channels
        ]
        return {"agents": agent_info, "channels": channel_info}
