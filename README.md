# Quantum MCAGI — No-LLM Collective Intelligence

> AI built on quantum physics — no LLM required.

Quantum MCAGI (Multi-Component Artificial General Intelligence) is a
multi-agent reasoning system that uses quantum-inspired mathematics instead
of large language models.  Each agent maintains a quantum state and makes
decisions through wave-function collapse, weighted by a fitness function
derived from observations of its environment.

## How It Works

| Concept | Classical AI (LLM) | Quantum MCAGI |
|---|---|---|
| Decision making | Token prediction | Wave-function collapse |
| Coordination | Message passing | Quantum entanglement |
| Exploration | Temperature sampling | Superposition |
| Inference | Forward pass | Quantum measurement |

### Architecture

```
Orchestrator
├── Agent α  ←──entangle──→  Agent β
├── Agent β  ←──entangle──→  Agent γ
└── Agent γ
```

1. **QuantumState** — normalised complex amplitude vector (2ⁿ dimensions)
2. **WaveFunction** — encodes possible decisions as amplitudes
3. **EntanglementChannel** — correlates multiple agents' states
4. **Measurement** — collapses state to a definite outcome
5. **QuantumAgent** — perceive → decide → act loop
6. **Orchestrator** — multi-agent coordination and consensus

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Bonsib420/REPLIT-QUANTUM-MCAGI-NO-LLM.git
cd REPLIT-QUANTUM-MCAGI-NO-LLM

# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py

# Run the tests
pip install pytest
pytest
```

### Run on Replit

Click **Run** — the `.replit` and `replit.nix` files are already configured.

## Project Structure

```
.
├── main.py                    # Entry point / demo
├── quantum_mcagi/
│   ├── __init__.py
│   ├── quantum_state.py       # Amplitude vector & collapse
│   ├── wave_function.py       # Decision superposition
│   ├── entanglement.py        # Agent correlation channel
│   ├── measurement.py         # Measurement & entropy
│   ├── agent.py               # Single quantum agent
│   └── orchestrator.py        # Multi-agent orchestrator
├── tests/
│   ├── test_quantum_state.py
│   ├── test_wave_function.py
│   ├── test_agent.py
│   └── test_orchestrator.py
├── .replit                    # Replit run configuration
├── replit.nix                 # Replit Nix environment
├── pyproject.toml             # Python project metadata
├── requirements.txt           # pip dependencies
└── REPLIT_MIGRATION.md        # How to sync Replit ↔ GitHub
```

## Replit ↔ GitHub Sync

See **[REPLIT_MIGRATION.md](REPLIT_MIGRATION.md)** for step-by-step
instructions on connecting your Replit project to this GitHub repository.

## License

MIT
