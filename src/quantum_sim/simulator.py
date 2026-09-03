import numpy as np

from .gates import I


class QuantumSimulator:
    def __init__(self, num_qubits: int):
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")

        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0

    def apply_single_qubit_gate(
        self,
        gate: np.ndarray,
        target: int,
    ) -> None:
        if target < 0 or target >= self.num_qubits:
            raise ValueError("Invalid target qubit")

        operator = np.array([[1]], dtype=complex)

        for qubit in range(self.num_qubits):
            if qubit == target:
                operator = np.kron(operator, gate)
            else:
                operator = np.kron(operator, I)

        self.state = operator @ self.state

    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2