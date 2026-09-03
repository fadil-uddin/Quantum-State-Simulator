import numpy as np
import pytest

from quantum_sim.gates import H, X
from quantum_sim.simulator import QuantumSimulator


def test_initial_state_one_qubit():
    sim = QuantumSimulator(1)

    expected = np.array([1, 0], dtype=complex)

    assert np.allclose(sim.state, expected)


def test_initial_state_two_qubits():
    sim = QuantumSimulator(2)

    expected = np.array([1, 0, 0, 0], dtype=complex)

    assert np.allclose(sim.state, expected)


def test_x_gate_on_one_qubit():
    sim = QuantumSimulator(1)

    sim.apply_single_qubit_gate(X, 0)

    expected = np.array([0, 1], dtype=complex)

    assert np.allclose(sim.state, expected)


def test_hadamard_on_one_qubit():
    sim = QuantumSimulator(1)

    sim.apply_single_qubit_gate(H, 0)

    expected = np.array(
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        dtype=complex,
    )

    assert np.allclose(sim.state, expected)


def test_probabilities_sum_to_one():
    sim = QuantumSimulator(1)

    sim.apply_single_qubit_gate(H, 0)

    assert np.isclose(np.sum(sim.probabilities()), 1.0)


def test_invalid_num_qubits():
    with pytest.raises(ValueError):
        QuantumSimulator(0)


def test_invalid_target_qubit():
    sim = QuantumSimulator(2)

    with pytest.raises(ValueError):
        sim.apply_single_qubit_gate(X, 2)