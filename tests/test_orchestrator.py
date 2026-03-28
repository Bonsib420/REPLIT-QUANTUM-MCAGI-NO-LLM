"""Tests for quantum_mcagi.orchestrator."""

import numpy as np

from quantum_mcagi.orchestrator import Orchestrator


class TestOrchestrator:
    def test_add_agent(self):
        orch = Orchestrator()
        a = orch.add_agent("a")
        assert len(orch.agents) == 1
        assert a.name == "a"

    def test_entangle_creates_channel(self):
        orch = Orchestrator()
        a = orch.add_agent("a")
        b = orch.add_agent("b")
        ch = orch.entangle(a, b)
        assert len(orch.channels) == 1
        assert ch.correlation > 0

    def test_collective_decide(self):
        orch = Orchestrator()
        orch.add_agent("a")
        orch.add_agent("b")
        result = orch.collective_decide(["x", "y"], [0.9, 0.1])
        assert result["consensus"] in ("x", "y")
        assert "votes" in result

    def test_report(self):
        orch = Orchestrator()
        orch.add_agent("a")
        rep = orch.report()
        assert len(rep["agents"]) == 1
        assert "entropy" in rep["agents"][0]
