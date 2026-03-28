"""
main.py — Quantum MCAGI demonstration.

Runs a multi-agent quantum AI system through a simple resource-management
scenario entirely without large language models.  All decision-making is
performed via quantum state superposition, wave function interference, and
Born-rule measurement.

Usage
-----
    python main.py
"""

from __future__ import annotations

import numpy as np

from quantum_mcagi import (
    QuantumAgent,
    QuantumOrchestrator,
)


# ---------------------------------------------------------------------------
# Scenario: Quantum Resource Allocation
# ---------------------------------------------------------------------------
# Three agents collaboratively decide how to allocate a shared resource across
# time steps.  The environment provides a demand signal; an oracle amplifies
# actions that match demand; rewards are positive when supply meets demand.

ACTIONS = ["allocate_low", "allocate_medium", "allocate_high", "reallocate"]

DEMAND_LEVELS = ["low", "medium", "high"]


def demand_oracle(action: str, demand: float) -> float:
    """Score actions based on how well they match the current demand level.

    Returns a phase in [-π, π] used by the WaveFunction oracle.
    """
    action_levels = {
        "allocate_low": 0.25,
        "allocate_medium": 0.50,
        "allocate_high": 0.75,
        "reallocate": 0.50,
    }
    level = action_levels.get(action, 0.5)
    # Higher score when action level ≈ demand (normalised to [0,1])
    norm_demand = demand / 10.0
    match = 1.0 - abs(level - norm_demand)
    return (match - 0.5) * np.pi  # map to [-π/2, π/2]


def reward_fn(action: str, demand: float) -> float:
    """Reward for the chosen collective action."""
    action_levels = {
        "allocate_low": 2.5,
        "allocate_medium": 5.0,
        "allocate_high": 7.5,
        "reallocate": 5.0,
    }
    supply = action_levels.get(action, 5.0)
    over = max(0.0, supply - demand)
    under = max(0.0, demand - supply)
    return 1.0 - 0.1 * over - 0.3 * under


def generate_scenario(n_steps: int, rng: np.random.Generator) -> list[dict]:
    """Generate a sequence of observations with demand signals."""
    observations = []
    for t in range(n_steps):
        demand = float(rng.uniform(1, 9))
        trend = np.sin(t * 0.4) * 3.0 + 5.0
        observations.append(
            {
                "demand": demand,
                "trend": trend,
                "step": float(t),
                "noise": float(rng.normal(0, 0.2)),
            }
        )
    return observations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Quantum MCAGI — Multi-Context AGI (No LLM)")
    print("  Quantum-inspired multi-agent decision system")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)

    # Create three specialised agents
    agent_alpha = QuantumAgent(
        "Alpha",
        actions=ACTIONS,
        belief_dim=8,
        learning_rate=0.15,
        rng=rng,
    )
    agent_beta = QuantumAgent(
        "Beta",
        actions=ACTIONS,
        belief_dim=8,
        learning_rate=0.10,
        rng=rng,
    )
    agent_gamma = QuantumAgent(
        "Gamma",
        actions=ACTIONS,
        belief_dim=8,
        learning_rate=0.20,
        rng=rng,
    )

    orchestrator = QuantumOrchestrator(
        agents=[agent_alpha, agent_beta, agent_gamma],
        shared_belief_dim=16,
        rng=rng,
    )

    # Entangle Alpha and Beta: their beliefs become correlated
    orchestrator.entangle_agents(agent_alpha, agent_beta, correlation=0.7)
    print(f"\nEntanglement registry: {orchestrator.entanglement_registry}")

    # Generate a 20-step scenario
    observations = generate_scenario(n_steps=20, rng=rng)

    print("\n--- Running 20-step episode ---\n")
    total_reward = 0.0
    for obs in observations:
        demand = obs["demand"]
        collective = orchestrator.run_step(
            obs,
            oracle=lambda action, d=demand: demand_oracle(action, d),
            reward_fn=lambda action, d=demand: reward_fn(action, d),
        )
        reward = orchestrator.history[-1]["reward"]
        total_reward += reward
        agent_votes = orchestrator.history[-1]["agent_actions"]
        print(
            f"Step {orchestrator._step:2d} | demand={demand:.2f} | "
            f"votes={list(agent_votes.values())} | "
            f"→ collective={collective!r:20s} | reward={reward:+.3f}"
        )

    print(f"\n--- Episode complete | total_reward={total_reward:.3f} ---")

    print("\n--- Agent action probabilities (after learning) ---")
    for agent in orchestrator.agents:
        top = agent.decision_wf.top_k(2)
        items = ", ".join(f"{a}:{p:.3f}" for a, p in top)
        print(f"  {agent.name}: [{items}]")

    print("\n--- Orchestrator summary (last 5 steps) ---")
    print(orchestrator.summary())

    print("\n--- Quantum state introspection ---")
    print(f"  Alpha belief:  {agent_alpha.belief}")
    print(f"  Beta belief:   {agent_beta.belief}")
    print(f"  Gamma belief:  {agent_gamma.belief}")

    print("\nDone. ✓")


if __name__ == "__main__":
    main()
