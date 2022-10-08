import qiskit
import copy
import random
from functools import partial

from PatchedMeasCal.edge_bfs import CouplingMapGraph
from PatchedMeasCal.inv_measure_methods import strip_measurement

def jigsaw(circuit, backend, n_shots):

    n_qubits = len(measurement_results.keys().__iter__().__next__())

    n_shots_global = n_shots // 2
    n_shots_pmfs = n_shots // 2

    global_pmf_table = global_pmf(circuit, backend, n_shots)

    # Picking random pairs
    local_pmf_pairs = random.shuffle(list(range(n_qubits)))
    local_pmf_pairs = [
            [i, j] for i, j in zip(
                local_pmf_pairs[::2],
                local_pmf_pairs[1::2]
                )
        ]

    # Because qiskit stores results strings backwards, the index ordering is reversed
    local_pmf_pairs_index = [
            [n_qubits - i, n_qubits - j] for i, j in zip(
                local_pmf_pairs[::2],
                local_pmf_pairs[1::2]
                )
        ]

    local_pmf_circs = [local_pmf_circuit(circuit, backend, n_shots_pmfs, i) for i in local_pmf_pairs]
    local_pmf_tables = local_pmf_table(local_pmf_circs, backend)

    for table, pair in zip(local_pmf_tables, local_pmf_pairs_index):
        global_pmf_table = convolve(global_pmf_table, table, pair)

    return global_pmf_table


def local_pmf_circuit(circuit, backend, n_shots, targets):
    '''
        Builds a circuit for a local pmf
    '''
    stripped_circuit = strip_measurement(circuit)
    stripped_circuit.measure(targets, list(range(len(targets))))
    
    n_qubits = len(backend.properties()._qubits)
    qubit_layout = list(range(n_qubits))
    tc = qiskit.transpile(circuit, backend=backend, initial_layout=qubit_layout, optimization_level=0)
    return tc

def global_pmf(circuit, backend, n_shots):
    n_qubits = len(backend.properties()._qubits)
    qubit_layout = list(range(n_qubits))

    tc = qiskit.transpile(circuit, backend=backend, initial_layout=qubit_layout, optimization_level=0)
    res = qiskit.execute(tc, backend=backend, initial_layout=qubit_layout, optimization_level=0).results().get_counts()
    return tc

def local_pmf_tables(circs, pairs):
    n_qubits = len(backend.properties()._qubits)
    qubit_layout = list(range(n_qubits))
    return qiskit.execute(circs, backend=backend, initial_layout=qubit_layout, optimization_level=0).results().get_counts()


def convolve(global_pmf_table, local_table, local_pair):
    '''
        This is the Bayes update for the jigsaw
    '''

    # Build the table
    split_table = {bin(i)[2:].zfill(len(local_pair)):{} for i in range(2 ** len(local_pair))}
    for table_str in global_pmf_table:
        table_idx = ''.join(table_str[i] for i in local_pair)
        table_col = ''.join(table_chr for i, table_chr in enumerate(table_str) if i not in local_pair)
        split_table[table_idx][table_col] = global_pmf_table[table_str]

    # Normalise each row and update
    for idx in split_table:
        subtable = split_table[idx]
        norm_val = sum(subtable.values())
        for jdx in subtable:
            subtable[jdx] /= norm_val # Norm
            subtable[jdx] *= local_table[idx] / (1 - local_table[idx]) # Bayes Update

    # Rejoin the table
    joint_table = {}
    for idx in split_table:
        subtable = split_table[idx]
        for jdx in subtable:
            table_idx = jdx
            for i, pos in enumerate(local_pair):
                table_idx = table_idx[:pos] + idx[i] + table_idx[pos:]
            joint_table[table_idx] = subtable[jdx]

    # Normalise the final table
    norm_val = sum(joint_table.values())
    for i in joint_table:
        joint_table[i] /= norm_val
    return joint_table
