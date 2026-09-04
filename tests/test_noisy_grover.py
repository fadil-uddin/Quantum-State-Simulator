import numpy as np
import pytest

from quantum_sim.experiments import estimate_success_probability


def test_no_noise_gives_perfect_success():
    success_probability = estimate_success_probability(
        noise_type="bit_flip",
        noise_probability=0.0,
        trials=100,
        marked_state="10",
        seed=42,
    )

    assert np.isclose(success_probability, 1.0)


def test_bit_flip_noise_reduces_success():
    success_probability = estimate_success_probability(
        noise_type="bit_flip",
        noise_probability=0.5,
        trials=500,
        marked_state="10",
        seed=42,
    )

    assert success_probability < 1.0


def test_depolarizing_noise_reduces_success():
    success_probability = estimate_success_probability(
        noise_type="depolarizing",
        noise_probability=0.5,
        trials=500,
        marked_state="10",
        seed=42,
    )

    assert success_probability < 1.0


def test_success_probability_is_valid_probability():
    success_probability = estimate_success_probability(
        noise_type="bit_flip",
        noise_probability=0.25,
        trials=200,
        marked_state="10",
        seed=123,
    )

    assert 0.0 <= success_probability <= 1.0


def test_invalid_noise_type():
    with pytest.raises(ValueError):
        estimate_success_probability(
            noise_type="invalid",
            noise_probability=0.1,
            trials=10,
            marked_state="10",
            seed=42,
        )


def test_invalid_trial_count():
    with pytest.raises(ValueError):
        estimate_success_probability(
            noise_type="bit_flip",
            noise_probability=0.1,
            trials=0,
            marked_state="10",
            seed=42,
        )