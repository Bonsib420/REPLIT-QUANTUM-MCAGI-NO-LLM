# Quantum MCAGI — No LLM

**Quantum-inspired Multi-Context AGI built on quantum physics principles — no large language models.**

This project implements a multi-agent AI system whose reasoning is grounded in
quantum mechanics: superposition, wave functions, entanglement, and Born-rule
measurement.  Every decision emerges from quantum amplitude operations rather
than from transformer-based language models.

---

## Key Concepts

| Quantum concept | Role in MCAGI |
|---|---|
| **Superposition** | Agents hold multiple hypotheses/actions simultaneously |
| **Wave function** | Probability distribution over the decision space |
| **Entanglement** | Correlated beliefs between cooperating agents |
| **Measurement** | Collapsing the wave function to a concrete decision |
| **Interference** | Amplifying promising actions, suppressing poor ones |
| **Oracle (Grover)** | Domain knowledge encoded as phase kicks |

---

## Project Structure

```
.
├── main.py                        # Entry point / demo
├── requirements.txt               # Python dependencies
├── .replit                        # Replit run configuration
├── replit.nix                     # Replit Nix environment
└── quantum_mcagi/
    ├── __init__.py
    ├── quantum_state.py           # QuantumState — complex amplitude vector
    ├── wave_function.py           # WaveFunction — named decision space
    ├── entanglement.py            # EntangledPair & EntanglementRegistry
    ├── measurement.py             # Projective, weak, and repeated measurement
    ├── agent.py                   # QuantumAgent — perceive/deliberate/decide/learn
    └── orchestrator.py            # QuantumOrchestrator — multi-agent coordination
```

---

## Quick Start

### On Replit

Click **Run** — Replit will install dependencies and execute `main.py` automatically.

### Locally

```bash
# Clone the repo
git clone https://github.com/Bonsib420/REPLIT-QUANTUM-MCAGI-NO-LLM.git
cd REPLIT-QUANTUM-MCAGI-NO-LLM

# Install dependencies (only numpy & scipy required)
pip install -r requirements.txt

# Run the demo
python main.py
```

---

## How It Works

### 1. `QuantumState`
A normalised complex vector `|ψ⟩` of length `n_basis`.  Supports Hadamard
(equal superposition), phase rotations, real rotations, interference, and
projective collapse.

### 2. `WaveFunction`
Maps a `QuantumState` onto a named option space (actions, hypotheses, …).
Provides oracle-driven amplitude amplification (Grover-inspired) and
Born-rule sampling.

### 3. `EntangledPair` / `EntanglementRegistry`
Creates Bell-state-inspired correlations between two `QuantumState` objects.
Measuring one state partially collapses the other according to the joint
probability tensor.

### 4. `Measurement`
Three measurement strategies:
- **Projective** — strong measurement, full collapse.
- **Weak** — partial collapse with tunable strength.
- **Repeated** — non-destructive probability estimation via sampling.

### 5. `QuantumAgent`
Implements the core MCAGI loop:
1. **Perceive** — encode observations as phase kicks on the belief state.
2. **Deliberate** — apply oracle + learned biases to the decision wave function.
3. **Decide** — sample an action via measurement.
4. **Learn** — update amplitude biases based on reward feedback.

### 6. `QuantumOrchestrator`
Coordinates a population of `QuantumAgent`s:
- Distributes observations to all agents.
- Collects individual decisions and runs quantum voting (amplitude-weighted).
- Broadcasts rewards; supports agent entanglement.

---

## Dependencies

- **numpy** ≥ 1.24 — array operations and random sampling
- **scipy** ≥ 1.10 — (available for future extensions)

No LLMs, no transformer libraries, no GPU required.

---

## Extending the Project

- Implement new oracle functions for your domain.
- Add more agents with different `learning_rate` or `belief_dim`.
- Wire real environment observations (sensors, APIs) into `orchestrator.run_step()`.
- Swap the quantum voting mechanism for Bell-inequality-inspired aggregation.

