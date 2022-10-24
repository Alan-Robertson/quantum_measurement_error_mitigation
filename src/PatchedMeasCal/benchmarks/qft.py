import copy

import qiskit
import numpy as np

from PatchedMeasCal.utils import norm_results_dict, dict_distance

def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(np.pi / 2 ** (n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)


def swap_registers(circuit, n):
    for qubit in range(n // 2):
        circuit.swap(qubit, n - qubit - 1)
    return circuit

def qft_circuit(qft_val, n_qubits):
        circ = qiskit.QuantumCircuit(n_qubits, n_qubits)
        
        # Prep state
        for i in range(n_qubits):
            circ.h(n_qubits - i - 1)
            circ.p(qft_val * np.pi / (2 ** i), n_qubits - i - 1)
        
        # QFT
        qft_rotations(circ, n_qubits)
        swap_registers(circ, n_qubits)

        circ.measure(list(range(n_qubits)), list(range(n_qubits)))
        
        return circ




