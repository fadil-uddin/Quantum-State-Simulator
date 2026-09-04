import matplotlib.pyplot as plt
import numpy as np

from quantum_sim.experiments import estimate_success_probability


def main():
    noise_probabilities = np.linspace(0.0, 0.5, 11)

    bit_flip_success = []
    depolarizing_success = []

    for probability in noise_probabilities:
        bit_flip_result = estimate_success_probability(
            noise_type="bit_flip",
            noise_probability=probability,
        )

        depolarizing_result = estimate_success_probability(
            noise_type="depolarizing",
            noise_probability=probability,
        )

        bit_flip_success.append(bit_flip_result)
        depolarizing_success.append(depolarizing_result)

        print(
            f"p={probability:.2f} | "
            f"bit-flip={bit_flip_result:.3f} | "
            f"depolarizing={depolarizing_result:.3f}"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        noise_probabilities,
        bit_flip_success,
        marker="o",
        label="Bit-flip noise",
    )

    plt.plot(
        noise_probabilities,
        depolarizing_success,
        marker="s",
        label="Depolarizing noise",
    )

    plt.axhline(
        y=0.25,
        linestyle="--",
        label="Random guess baseline",
    )

    plt.xlabel("Noise probability")
    plt.ylabel("Grover success probability")
    plt.title("Effect of Noise on Two-Qubit Grover Search")

    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "grover_noise_comparison.png",
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":
    main()