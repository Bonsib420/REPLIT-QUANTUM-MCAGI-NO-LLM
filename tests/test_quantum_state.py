"""Tests for quantum_mcagi.quantum_state."""

import numpy as np
import pytest

from quantum_mcagi.quantum_state import QuantumState


class TestQuantumStateInit:
    def test_normalises_amplitudes(self):
        qs = QuantumState([1, 1])
        assert np.isclose(np.linalg.norm(qs.amplitudes), 1.0)

    def test_rejects_zero_vector(self):
        with pytest.raises(ValueError, match="zero vector"):
            QuantumState([0, 0])

    def test_rejects_non_power_of_two(self):
        with pytest.raises(ValueError, match="power of 2"):
            QuantumState([1, 1, 1])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            QuantumState([])


class TestBasisAndUniform:
    def test_basis_state(self):
        qs = QuantumState.basis(0, 2)
        assert qs.dimension == 4
        assert np.isclose(qs.probabilities[0], 1.0)

    def test_uniform_state(self):
        qs = QuantumState.uniform(2)
        assert np.allclose(qs.probabilities, 0.25)


class TestApply:
    def test_identity_preserves_state(self):
        qs = QuantumState.basis(0, 1)
        identity = np.eye(2)
        new_qs = qs.apply(identity)
        assert np.allclose(new_qs.amplitudes, qs.amplitudes)

    def test_not_gate_flips(self):
        qs = QuantumState.basis(0, 1)
        x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
        new_qs = qs.apply(x_gate)
        assert np.isclose(new_qs.probabilities[1], 1.0)


class TestCollapse:
    def test_collapse_returns_valid_index(self):
        qs = QuantumState.uniform(2)
        rng = np.random.default_rng(42)
        idx = qs.collapse(rng=rng)
        assert 0 <= idx < 4

    def test_basis_state_always_collapses_to_itself(self):
        qs = QuantumState.basis(2, 2)
        for _ in range(20):
            assert qs.collapse() == 2
