import numpy as np

from .gates import H
from .simulator import QuantumSimulator


def apply_oracle(sim: QuantumSimulator, marked_state: str) -> None:
    if len(marked_state) != sim.num_qubits:
        raise ValueError("Marked state length must match number of qubits")

    if any(bit not in {"0", "1"} for bit in marked_state):
        raise ValueError("Marked state must be a binary string")

    marked_index = int(marked_state, 2)
    sim.state[marked_index] *= -1


def apply_diffusion(sim: QuantumSimulator) -> None:
    size = 2 ** sim.num_qubits

    uniform_state = np.ones(size, dtype=complex) / np.sqrt(size)

    diffusion_operator = (
        2 * np.outer(uniform_state, uniform_state.conj())
        - np.eye(size, dtype=complex)
    )

    sim.state = diffusion_operator @ sim.state


def run_grover(
    num_qubits: int,
    marked_state: str,
    iterations: int = 1,
) -> QuantumSimulator:
    sim = QuantumSimulator(num_qubits)

    for qubit in range(num_qubits):
        sim.apply_single_qubit_gate(H, qubit)

    for _ in range(iterations):
        apply_oracle(sim, marked_state)
        apply_diffusion(sim)

    return sim