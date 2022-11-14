import qiskit
import copy
import random
from functools import partial

from PatchedMeasCal.edge_bfs import CouplingMapGraph
from PatchedMeasCal.inv_measure_methods import strip_measurement
from PatchedMeasCal.utils import norm_results_dict

def jigsaw(circuit, backend, n_shots, 
    verbose=False,  # Verbosity
    equal_shot_distribution=False, # Splits number of shots equally, otherwise uses that number of shots per experiment
    local_pmf_pairs=None, # Local pairs, random if not set
    probs=None, # False error prob distribution
    n_qubits=None,  # Number of qubits
    meas_filter=None, # Inbuilt meas filter pass
    norm_fix=False # Our normalisation fix for JIGSAW
    ):

    if n_qubits is None and backend.properties() is not None:
        n_qubits = len(backend.properties()._qubits)

    global_pmf_table = build_global_pmf(circuit, backend, n_shots, probs=probs, n_qubits=n_qubits)

    # If no allocation is provided, use random
    if local_pmf_pairs is None:
        local_pmf_pairs = list(range(n_qubits))
        random.shuffle(local_pmf_pairs)
        local_pmf_pairs = [
                [i, j] for i, j in zip(
                    local_pmf_pairs[::2],
                    local_pmf_pairs[1::2]
                    )
            ]

    if equal_shot_distribution:
        n_shots_global = n_shots // 2
        n_shots_pmfs = n_shots // (2 * len(local_pmf_pairs))
    else:
        n_shots_global = n_shots
        n_shots_pmfs = n_shots

    # Because qiskit stores results strings backwards, the index ordering is reversed
    local_pmf_pairs_index = [[(n_qubits - i) % n_qubits, (n_qubits - j) % n_qubits] for i, j in local_pmf_pairs]

    local_pmf_circs = [build_local_pmf_circuit(circuit, backend, pairs, n_qubits=n_qubits) for pairs in local_pmf_pairs]

    local_pmf_tables = build_local_pmf_tables(
        local_pmf_circs,
        local_pmf_pairs,
        backend,
        n_shots_pmfs,
        probs=probs,
        n_qubits=n_qubits)

    for table, pair in zip(local_pmf_tables, local_pmf_pairs[::-1]):
        global_pmf_table = convolve(global_pmf_table, table, pair, norm_fix=norm_fix)

    return global_pmf_table


def build_local_pmf_circuit(circuit, backend, targets, n_qubits=None):
    '''
        Builds a circuit for a local pmf
    '''
    if n_qubits is None:
        n_qubits = len(backend.properties()._qubits)
    qubit_layout = list(range(n_qubits))
    tc = qiskit.transpile(circuit, backend=backend, initial_layout=qubit_layout, optimization_level=0)

    stripped_circuit = strip_measurement(tc)
    stripped_circuit.cregs = [stripped_circuit._create_creg(len(targets), 'c')]
    stripped_circuit.measure(targets, list(range(len(targets))))
    return stripped_circuit

def build_local_pmf_tables(circs, pairs, backend, n_shots, probs=None, n_qubits=None, meas_filter=None):

    if n_qubits is None:
        n_qubits = len(backend.properties()._qubits)
    qubit_layout = list(range(n_qubits))
    local_pmf_tables = qiskit.execute(circs, 
        backend=backend, 
        initial_layout=qubit_layout, 
        optimization_level=0,
        shots=n_shots).result().get_counts()

    if meas_filter is not None:
        for i, pair in enumerate(pairs):
            local_pmf_tables[i] = meas_filter(local_pmf_tables[i], participating_qubits=pair)

    for i, table in enumerate(local_pmf_tables):
        if probs is not None:
            pair_probs = probs.sub_set(len(pairs[i]), participating_qubits=pairs[i])
            local_pmf_tables[i] = pair_probs(local_pmf_tables[i])
        norm_results_dict(local_pmf_tables[i])

    return local_pmf_tables

def build_global_pmf(circuit, backend, n_shots, probs=None, n_qubits=None):
    if n_qubits is None:
        n_qubits = len(backend.properties()._qubits)
    qubit_layout = list(range(n_qubits))

    tc = qiskit.transpile(circuit, backend=backend, initial_layout=qubit_layout, optimization_level=0)
    res = qiskit.execute(tc, backend=backend, initial_layout=qubit_layout, optimization_level=0).result().get_counts()
    
    if probs is not None:
        res = probs(res)

    norm_results_dict(res)
    return res


def convolve(global_pmf_table, local_table, local_pair, norm_fix=False):
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
            if norm_fix and subtable[jdx] < norm_val: # Resolves issue of single element entries norming to 1
                subtable[jdx] /= norm_val 
            elif not norm_fix:
                subtable[jdx] /= norm_val 
            if idx in local_table: # This can happen
                if local_table[idx] == 1: # This can also happen, we're just avoiding a div 0 here
                    local_table[idx] == 0.9999
                try:
                    subtable[jdx] *= local_table[idx] / (1 - local_table[idx]) # Bayes Update
                except:
                    pass

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
    norm_results_dict(joint_table)
    return joint_table
