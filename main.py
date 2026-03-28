#!/usr/bin/env python3
"""Quantum MCAGI — AI built on quantum physics, no LLM.

Run this script to see the multi-agent system in action.
"""

import json

import numpy as np

from quantum_mcagi import Orchestrator


def main() -> None:
    print("=" * 60)
    print("  Quantum MCAGI — No-LLM Collective Intelligence")
    print("=" * 60)

    # --- Bootstrap agents ---
    orch = Orchestrator()
    alpha = orch.add_agent("Alpha", n_qubits=3)
    beta = orch.add_agent("Beta", n_qubits=3)
    gamma = orch.add_agent("Gamma", n_qubits=3)

    # --- Entangle agents for coordinated reasoning ---
    orch.entangle(alpha, beta)
    orch.entangle(beta, gamma)

    # --- Feed an observation (simulated sensor data) ---
    observation = np.array([0.1, 0.5, 0.9, 0.2, 0.7, 0.3, 0.8, 0.4])
    orch.broadcast_observation(observation)

    # --- Synchronise entangled agents ---
    orch.synchronise()

    # --- Collective decision ---
    options = ["explore", "exploit", "rest", "communicate"]
    fitness = [0.6, 0.9, 0.2, 0.7]
    result = orch.collective_decide(options, fitness)

    print("\n-- Collective Decision --")
    print(f"  Consensus : {result['consensus']}")
    print(f"  Votes     : {result['votes']}")

    for detail in result["agent_details"]:
        print(f"\n  Agent {detail['agent']}:")
        print(f"    chose   → {detail['decision']}")
        dist = detail["measurement_distribution"]
        top = sorted(dist.items(), key=lambda x: -x[1])[:3]
        print(f"    top-3   → {dict(top)}")

    # --- System report ---
    print("\n-- System Report --")
    report = orch.report()
    print(json.dumps(report, indent=2))

    print("\n✓ Quantum MCAGI run complete.")


if __name__ == "__main__":
    main()
