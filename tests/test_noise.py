import numpy as np
import pytest

from quantum_sim.gates import H, X
from quantum_sim.noise import (
    apply_bit_flip_noise,
    apply_depolarizing_noise,
)
from quantum_sim.simulator import QuantumSimulator


def test_bit_flip_noise_probability_zero():
    sim = QuantumSimulator(1)

    apply_bit_flip_noise(sim, probability=0.0)

    expected = np.array([1, 0], dtype=complex)

    assert np.allclose(sim.state, expected)


def test_bit_flip_noise_probability_one():
    sim = QuantumSimulator(1)

    apply_bit_flip_noise(sim, probability=1.0)

    expected = np.array([0, 1], dtype=complex)

    assert np.allclose(sim.state, expected)


def test_bit_flip_noise_two_qubits_probability_one():
    sim = QuantumSimulator(2)

    apply_bit_flip_noise(sim, probability=1.0)

    expected = np.array([0, 0, 0, 1], dtype=complex)

    assert np.allclose(sim.state, expected)


def test_invalid_bit_flip_probability():
    sim = QuantumSimulator(1)

    with pytest.raises(ValueError):
        apply_bit_flip_noise(sim, probability=-0.1)

    with pytest.raises(ValueError):
        apply_bit_flip_noise(sim, probability=1.1)


def test_depolarizing_noise_probability_zero():
    sim = QuantumSimulator(1)

    apply_depolarizing_noise(sim, probability=0.0)

    expected = np.array([1, 0], dtype=complex)

    assert np.allclose(sim.state, expected)


def test_invalid_depolarizing_probability():
    sim = QuantumSimulator(1)

    with pytest.raises(ValueError):
        apply_depolarizing_noise(sim, probability=-0.1)

    with pytest.raises(ValueError):
        apply_depolarizing_noise(sim, probability=1.1)


def test_depolarizing_noise_preserves_normalisation():
    sim = QuantumSimulator(1)

    sim.apply_single_qubit_gate(H, 0)

    rng = np.random.default_rng(42)

    apply_depolarizing_noise(
        sim,
        probability=1.0,
        rng=rng,
    )

    probability_sum = np.sum(sim.probabilities())

    assert np.isclose(probability_sum, 1.0)


def test_bit_flip_noise_reproducible_with_seed():
    sim_a = QuantumSimulator(3)
    sim_b = QuantumSimulator(3)

    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    apply_bit_flip_noise(
        sim_a,
        probability=0.5,
        rng=rng_a,
    )

    apply_bit_flip_noise(
        sim_b,
        probability=0.5,
        rng=rng_b,
    )

    assert np.allclose(sim_a.state, sim_b.state)