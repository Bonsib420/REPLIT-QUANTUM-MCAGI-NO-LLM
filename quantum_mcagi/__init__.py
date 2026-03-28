"""
Quantum MCAGI — Quantum-inspired Multi-Context AGI without LLMs.

This package implements an AI system grounded in quantum physics principles:
superposition, wave functions, entanglement, and measurement.  All reasoning
and decision-making is performed through quantum-inspired algorithms rather
than large language models.
"""

from .quantum_state import QuantumState
from .wave_function import WaveFunction
from .entanglement import EntangledPair, EntanglementRegistry
from .measurement import Measurement
from .agent import QuantumAgent
from .orchestrator import QuantumOrchestrator

__all__ = [
    "QuantumState",
    "WaveFunction",
    "EntangledPair",
    "EntanglementRegistry",
    "Measurement",
    "QuantumAgent",
    "QuantumOrchestrator",
]
