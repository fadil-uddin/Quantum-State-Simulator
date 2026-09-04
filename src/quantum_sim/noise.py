import numpy as np

from .gates import I, X, Y, Z
from .simulator import QuantumSimulator


def _validate_probability(probability: float) -> None:
    if probability < 0.0 or probability > 1.0:
        raise ValueError("Noise probability must be between 0 and 1")


def apply_bit_flip_noise(
    sim: QuantumSimulator,
    probability: float,
    rng: np.random.Generator | None = None,
) -> None:
    _validate_probability(probability)

    if rng is None:
        rng = np.random.default_rng()

    for qubit in range(sim.num_qubits):
        if rng.random() < probability:
            sim.apply_single_qubit_gate(X, qubit)


def apply_depolarizing_noise(
    sim: QuantumSimulator,
    probability: float,
    rng: np.random.Generator | None = None,
) -> None:
    _validate_probability(probability)

    if rng is None:
        rng = np.random.default_rng()

    pauli_errors = (X, Y, Z)

    for qubit in range(sim.num_qubits):
        if rng.random() < probability:
            error_gate = pauli_errors[rng.integers(0, len(pauli_errors))]
            sim.apply_single_qubit_gate(error_gate, qubit)