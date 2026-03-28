"""Tests for quantum_mcagi.agent."""

import numpy as np

from quantum_mcagi.agent import QuantumAgent


class TestQuantumAgent:
    def test_initial_state_is_uniform(self):
        agent = QuantumAgent(name="test", n_qubits=2)
        assert np.allclose(agent.state.probabilities, 0.25)

    def test_perceive_changes_state(self):
        agent = QuantumAgent(name="test", n_qubits=1)
        before = agent.state.amplitudes.copy()
        agent.perceive(np.array([0.5, 1.0]))
        after = agent.state.amplitudes
        assert not np.allclose(before, after)

    def test_decide_returns_valid_option(self):
        agent = QuantumAgent(name="test", n_qubits=2)
        d = agent.decide(["up", "down"], [1.0, 0.5])
        assert d.label in ("up", "down")

    def test_act_returns_summary(self):
        agent = QuantumAgent(name="test", n_qubits=2)
        result = agent.act(["a", "b"], [0.8, 0.2])
        assert "agent" in result
        assert "decision" in result
        assert result["decision"] in ("a", "b")
