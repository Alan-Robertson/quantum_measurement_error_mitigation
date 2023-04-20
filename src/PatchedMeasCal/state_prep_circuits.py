import qiskit
import numpy as np
import scipy

def res_to_sparse_vec(measurement_results):
    '''
        Helper function to rescale measurement results
    '''
    rows = list(map(lambda x: int(x[::-1], 2), measurement_results.keys()))
    cols = np.zeros(len(measurement_results), dtype=np.int32)
    values = list(measurement_results.values())

    # Faster to query the first element than it is to build the list
    n_qubits = len(measurement_results.keys().__iter__().__next__()) 
    n_shots = np.sum(values) 

    # Sparse construction
    results_vec = scipy.sparse.csc_array((values, (rows, cols)), shape=(2 ** n_qubits, 1), dtype=np.float32)
    return results_vec

def res_to_vec(measurement_results):
    '''
        Helper function to rescale measurement results
    '''

    n_qubits = len(measurement_results.keys().__iter__().__next__())
    n_elements = 2 ** n_qubits 
    results_vec = np.zeros(n_elements, dtype=np.float32)

    for state in measurement_results:
        results_vec[int(state[::-1], 2)] = measurement_results[state]
    return results_vec

def GHZ_prep(*args, **kwargs):
    return plus_state_prep(*args, **kwargs)

def plus_state_prep(backend, target_qubits = None):
    '''
        Circuit that prepares the |00...0> + |11...1> state
    '''
    n_qubits = len(backend.properties()._qubits)
    coupling_map = backend.configuration().coupling_map
    
    targeted_qubits = [0]
    circuit = qiskit.QuantumCircuit(n_qubits, n_qubits)
    circuit.h(0)
    
    # Greedy BFS, could be much better
    while len(targeted_qubits) < n_qubits:
        for i in range(n_qubits):
            if i not in targeted_qubits: # Qubit not yet added
                for edge in coupling_map: # For each edge in the map
                    if i in edge:
                        ctrl = edge[edge[0] == i]
                        targ = edge[edge[1] == i]
                        circuit.cnot(ctrl, targ)
                        targeted_qubits.append(i)
                        break
    circuit.measure(list(range(n_qubits)), list(range(n_qubits)))
    return circuit

def GHZ_state_dist(results:dict):
    return plus_state_dist(results)

def plus_state_dist(results:dict):
    n_qubits = len(results.keys().__iter__().__next__())

    dist = 0
    target_states = ['0' * n_qubits, '1' *  n_qubits]
    n_shots = sum(results.values())

    for state_str in target_states:
        if state_str in results:
            dist += np.abs(0.5 - (results[state_str] / n_shots))

    return dist



def equal_superposition_state_prep(backend):
    '''
        Circuit that prepares the |0> + |1> + ... + |2^n - 1> state
    '''
    n_qubits = len(backend.properties()._qubits)
    coupling_map = backend.configuration().coupling_map
    
    circuit = qiskit.QuantumCircuit(n_qubits, n_qubits)
    for i in range(n_qubits):
        circuit.h(i)
    circuit.measure(list(range(n_qubits)), list(range(n_qubits)))
    return circuit 

def equal_superposition_state_dist(results:dict, *args, **kwargs):
    n_qubits = len(results.keys().__iter__().__next__())
    vec = np.array(list(results.values())) / sum(list(results.values()))
    target_val = 1 / (2 ** n_qubits) 
    dist = np.sum(np.abs(vec - target_val))
    return dist

def integer_state_prep(backend, int_val):
    '''
        Circuit that prepares the |i> state
    '''
    n_qubits = len(backend.properties()._qubits)
    coupling_map = backend.configuration().coupling_map
    
    circuit = qiskit.QuantumCircuit(n_qubits, n_qubits)
    for i, val in zip(range(n_qubits), map(int, bin(int_val)[2:])):
        if val:
            circuit.x(i)
    circuit.measure(list(range(n_qubits)), list(range(n_qubits)))
    return circuit 

def integer_state_prep_sim(int_val, n_qubits):
    '''
        Circuit that prepares the |i> state
    '''    
    circuit = qiskit.QuantumCircuit(n_qubits, n_qubits)
    for i, val in zip(range(n_qubits), bin(int_val)[2:].zfill(n_qubits)):
        if val == '1':
            circuit.x(i)
    circuit.measure(list(range(n_qubits)), list(range(n_qubits)))
    return circuit 

def integer_state_dist(results:dict, int_val):
    n_qubits = len(results.keys().__iter__().__next__())
    int_val_key = bin(int_val)[2:].zfill(n_qubits)[::-1]

    dist = 1
    if int_val_key in results:
        dist = 1 - results[int_val_key] / sum(results.values())
    return dist

