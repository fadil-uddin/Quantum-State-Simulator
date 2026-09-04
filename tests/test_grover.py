import numpy as np
import pytest

from quantum_sim.grover import (
    apply_diffusion,
    apply_oracle,
    run_grover,
)
from quantum_sim.simulator import QuantumSimulator


def test_oracle_flips_marked_state_phase():
    sim = QuantumSimulator(2)

    sim.state = np.array(
        [0.5, 0.5, 0.5, 0.5],
        dtype=complex,
    )

    apply_oracle(sim, "10")

    expected = np.array(
        [0.5, 0.5, -0.5, 0.5],
        dtype=complex,
    )

    assert np.allclose(sim.state, expected)


def test_invalid_marked_state_length():
    sim = QuantumSimulator(2)

    with pytest.raises(ValueError):
        apply_oracle(sim, "1")


def test_invalid_marked_state_characters():
    sim = QuantumSimulator(2)

    with pytest.raises(ValueError):
        apply_oracle(sim, "2A")


def test_diffusion_preserves_normalisation():
    sim = QuantumSimulator(2)

    sim.state = np.array(
        [0.5, -0.5, 0.5, 0.5],
        dtype=complex,
    )

    apply_diffusion(sim)

    assert np.isclose(
        np.sum(np.abs(sim.state) ** 2),
        1.0,
    )


def test_two_qubit_grover_finds_marked_state():
    sim = run_grover(
        num_qubits=2,
        marked_state="10",
        iterations=1,
    )

    probabilities = sim.probabilities()

    assert np.isclose(probabilities[2], 1.0)