from quantum_sim.gates import H
from quantum_sim.simulator import QuantumSimulator


def main():
    sim = QuantumSimulator(2)

    sim.apply_single_qubit_gate(H, 0)
    sim.apply_cnot(control=0, target=1)

    print("Bell state amplitudes:")
    print(sim.state)

    print("\nProbabilities:")
    print(sim.probabilities())

    print("\nMeasurement:")
    print(sim.measure())


if __name__ == "__main__":
    main()