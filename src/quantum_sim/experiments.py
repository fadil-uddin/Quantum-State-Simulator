import numpy as np

from .grover import run_grover
from .noise import (
    apply_bit_flip_noise,
    apply_depolarizing_noise,
)


def estimate_success_probability(
    noise_type: str,
    noise_probability: float,
    trials: int = 1000,
    marked_state: str = "10",
    seed: int = 42,
) -> float:
    if trials < 1:
        raise ValueError("trials must be at least 1")

    rng = np.random.default_rng(seed)
    successes = 0

    for _ in range(trials):
        sim = run_grover(
            num_qubits=len(marked_state),
            marked_state=marked_state,
            iterations=1,
        )

        if noise_type == "bit_flip":
            apply_bit_flip_noise(
                sim,
                probability=noise_probability,
                rng=rng,
            )

        elif noise_type == "depolarizing":
            apply_depolarizing_noise(
                sim,
                probability=noise_probability,
                rng=rng,
            )

        else:
            raise ValueError(
                "noise_type must be 'bit_flip' or 'depolarizing'"
            )

        if sim.measure() == marked_state:
            successes += 1

    return successes / trials