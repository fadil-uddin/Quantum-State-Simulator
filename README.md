# Quantum State Simulator

A NumPy-based quantum state-vector simulator built from scratch.

This project implements core quantum-computing concepts without relying on a quantum SDK such as Qiskit or Cirq. It supports single- and two-qubit operations, measurement, Bell-state generation, Grover's search algorithm, simple noise models, and numerical experiments showing how noise affects quantum search performance.

## Features

- State-vector simulation for arbitrary numbers of qubits
- Single-qubit gates:
  - Identity
  - Pauli-X
  - Pauli-Y
  - Pauli-Z
  - Hadamard
- CNOT gate
- Computational-basis measurement with state collapse
- Bell-state generation
- Grover search
- Bit-flip noise
- Depolarizing noise
- Monte Carlo simulation of noisy Grover search
- Automated tests with pytest
- Visualisation of Grover success probability under noise

## Project Structure

```text
quantum-state-simulator/
│
├── examples/
│   ├── bell_state.py
│   ├── grover_search.py
│   └── noisy_grover.py
│
├── src/
│   └── quantum_sim/
│       ├── __init__.py
│       ├── gates.py
│       ├── grover.py
│       ├── noise.py
│       └── simulator.py
│
├── tests/
│   ├── test_gates.py
│   ├── test_grover.py
│   ├── test_noise.py
│   ├── test_noisy_grover.py
│   └── test_simulator.py
│
├── grover_noise_comparison.png
├── pyproject.toml
├── requirements.txt
└── README.md