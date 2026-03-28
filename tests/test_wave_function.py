"""Tests for quantum_mcagi.wave_function."""

import numpy as np
import pytest

from quantum_mcagi.wave_function import WaveFunction


class TestWaveFunction:
    def test_resolve_returns_decision(self):
        wf = WaveFunction()
        wf.add("a")
        wf.add("b")
        d = wf.resolve([1.0, 0.0], rng=np.random.default_rng(0))
        assert d.label == "a"

    def test_resolve_length_mismatch_raises(self):
        wf = WaveFunction()
        wf.add("x")
        with pytest.raises(ValueError, match="fitness length"):
            wf.resolve([1.0, 2.0])

    def test_all_zero_fitness_falls_back(self):
        wf = WaveFunction()
        wf.add("only")
        d = wf.resolve([0.0], rng=np.random.default_rng(0))
        assert d.label == "only"
