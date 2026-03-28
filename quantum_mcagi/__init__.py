"""Quantum MCAGI — AI built on quantum physics, no LLM."""

__version__ = "0.1.0"

from quantum_mcagi.quantum_state import QuantumState
from quantum_mcagi.wave_function import WaveFunction
from quantum_mcagi.entanglement import EntanglementChannel
from quantum_mcagi.measurement import measure
from quantum_mcagi.agent import QuantumAgent
from quantum_mcagi.orchestrator import Orchestrator

__all__ = [
    "QuantumState",
    "WaveFunction",
    "EntanglementChannel",
    "measure",
    "QuantumAgent",
    "Orchestrator",
]
