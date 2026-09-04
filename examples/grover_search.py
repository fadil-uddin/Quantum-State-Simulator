from quantum_sim.grover import run_grover


def main():
    marked_state = "10"

    sim = run_grover(
        num_qubits=2,
        marked_state=marked_state,
        iterations=1,
    )

    print(f"Marked state: {marked_state}")

    print("\nFinal amplitudes:")
    print(sim.state)

    print("\nFinal probabilities:")
    print(sim.probabilities())

    print("\nMeasurement:")
    print(sim.measure())


if __name__ == "__main__":
    main()