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

    def apply_cnot(self, control: int, target: int) -> None:
        if control < 0 or control >= self.num_qubits:
            raise ValueError("Invalid control qubit")

        if target < 0 or target >= self.num_qubits:
            raise ValueError("Invalid target qubit")

        if control == target:
            raise ValueError("Control and target qubits must be different")

        new_state = np.zeros_like(self.state)

        for index, amplitude in enumerate(self.state):
            bits = list(format(index, f"0{self.num_qubits}b"))

            if bits[control] == "1":
                bits[target] = "0" if bits[target] == "1" else "1"

            new_index = int("".join(bits), 2)
            new_state[new_index] += amplitude

        self.state = new_state