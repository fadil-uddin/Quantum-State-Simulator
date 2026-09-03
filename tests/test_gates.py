import numpy as np

from quantum_sim.gates import I, X, Y, Z, H


def test_identity_gate():
    expected = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=complex,
    )

    assert np.allclose(I, expected)


def test_x_gate():
    expected = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )

    assert np.allclose(X, expected)


def test_y_gate():
    expected = np.array(
        [
            [0, -1j],
            [1j, 0],
        ],
        dtype=complex,
    )

    assert np.allclose(Y, expected)


def test_z_gate():
    expected = np.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=complex,
    )

    assert np.allclose(Z, expected)


def test_hadamard_is_unitary():
    identity = np.eye(2, dtype=complex)

    assert np.allclose(H.conj().T @ H, identity)