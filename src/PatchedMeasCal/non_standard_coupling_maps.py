import qiskit
import numpy as np

from PatchedMeasCal.utils import list_fold
from itertools import combinations


def build_circuits(n_qubits):
    '''
        Builds basis circuits
    '''
    circs = [qiskit.QuantumCircuit(n_qubits, n_qubits) for i in range(2 ** n_qubits)]

    for i, circ in enumerate(circs):
        for j, val in enumerate(bin(i)[2:].zfill(n_qubits)[::-1]):
            if val == '1':
                circ.x(j)
        circ.measure(list(range(n_qubits)), list(range(n_qubits)))
    return circs

def qbnr2int(bnr):
    return int(bnr[::-1], 2)


def circuit_weight_coupling_map(circuit, disjoint=True):

    dag = qiskit.converters.circuit_to_dag(circuit)

    # Measured Nodes
    measured_qubits = [n.qargs[0]._index for n in dag.op_nodes() if n.name == 'measure']
    
    # Edge ordering
    edges = {}
    for edge in dag.two_qubit_ops():
        edge_qubits = [i._index for i in edge.qargs]
        edge_qubits.sort()
                
        edge_measured = True
        for i in edge_qubits:
            if i not in measured_qubits:
                edge_measured = False
                break
        
        if edge_measured:
            edge_key = str(edge_qubits)[1:-1]
            
            if edge_key in edges:
                edges[edge_key] += 1
            else:
                edges[edge_key] = 1

    # Sorted list of measured edges
    edges = list(edges.items())
    edges.sort(key = lambda x: x[1], reverse=True)

    # Drop the string indexing
    edges = [list(map(int, i[0].split(','))) for i in edges]   
    
    # Ensure disjoint edges
    if disjoint:
        disjoint_edges = []
        for edge in edges:
            if sum(i in measured_qubits for i in edge) == len(edge):
                disjoint_edges.append(edge)
                list(map(measured_qubits.remove, edge))
        edges = disjoint_edges
                
    # Add any dangling qubits to edges
    for edge in edges:
        for i in edge:
            if i in measured_qubits:
                measured_qubits.remove(i)
    for pair in list_fold(measured_qubits, 2):
        edges.append(pair)  
    
    return edges


def error_coupling_map(backend=None, k=5, error_profile=None, job_manager=qiskit.execute, n_shots=16000):
    
    if backend is not None:
        cal_res, pairs = build_calibrations(backend, k=k, job_manager=job_manager, n_shots=n_shots)
        error_profile = build_cal_matrices(cal_res, backend, pairs, k=k)

    n_qubits = len(error_profile['single']) 
    edge_weights = build_error_edges(error_profile, pairs)
    cmap_edges = edge_weights_to_cmap(edge_weights, n_qubits)
    return cmap_edges



def build_calibrations(backend, 
    k=5,
    job_manager=qiskit.execute,
    n_shots=16000):

    backend_cmap = qiskit.transpiler.CouplingMap(backend.configuration().coupling_map)
    n_qubits = len(backend.properties().qubits)
    layout = list(range(n_qubits))

    singles = layout
    pairs = list(combinations(layout, 2))

    i = 0
    while i < len(pairs):
        if backend_cmap.distance(*pairs[i]) > k:
            pairs.pop(i)
            i -= 1
        i += 1

    circuits_single = build_circuits(1)
    circuits_paired = build_circuits(2)

    tc = []
    for i in singles:
        tc += qiskit.transpile(circuits_single, backend=backend, initial_layout=[i])

    for i in pairs:
            tc += qiskit.transpile(circuits_paired, backend=backend, initial_layout=i)

    job = job_manager(tc, backend)
    return job.result(), pairs

def build_cal_matrices(results, backend, pairs, k=5):
    '''
        Given results and pairs build calibration matrices
    '''
    cal_matrices = {}

    n_qubits = len(backend.properties().qubits)
    singles = list(range(n_qubits))
    
    r_list = [results.get_counts(i) for i in range(len(results.results))]

    n_single = 2 * len(singles)
    n_pairs = 4 * len(pairs)

    single_cals = {}
    for targs, cal_res in zip(singles, list_fold(r_list[:n_single], 2)):
        res = np.zeros((2, 2), dtype=np.float32)
        for i, counts in enumerate(cal_res):
            c_vec = np.zeros(2, dtype=np.float32)
            for j in counts:
                c_vec[qbnr2int(j)] = counts[j]
            c_vec /= np.sum(c_vec)
            res[:,i] = c_vec

        single_cals[str(targs)] = res

    pair_cals = {}
    for targs, cal_res in zip(pairs, list_fold(r_list[n_single:], 4)):
        res = np.zeros((4, 4), dtype=np.float32)
        for i, counts in enumerate(cal_res):
            c_vec = np.zeros(4, dtype=np.float32)
            for j in counts:
                c_vec[qbnr2int(j)] = counts[j]
            c_vec /= np.sum(c_vec)
            idx = int(bin(i)[2:].zfill(2)[::-1], 2)
            res[:,idx] = c_vec

        pair_cals[str(targs)[1:-1]] = res   

    cal_matrices['single'] = single_cals
    cal_matrices['paired'] = pair_cals
    return cal_matrices
    



def build_error_edges(err_profile, pairs):
    n_qubits = len(err_profile['single'])
    edge_weights = np.zeros((n_qubits, n_qubits))

    n_samples = len(err_profile)

    single_cals = err_profile['single']
    pair_cals = err_profile['paired']

    for pair in pairs:

        pair_idx = "{}, {}".format(*pair)
        single_idx = list(map(str, pair))

        edge_weights[pair] += np.linalg.norm(
            pair_cals[pair_idx]
            - np.kron(single_cals[single_idx[0]], single_cals[single_idx[1]])
        )

    edge_weights /= n_samples
    return edge_weights

def edge_weights_to_cmap(edge_weights, n_qubits):

    error_weights_lst = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if edge_weights[i, j] > 0:
                error_weights_lst.append((edge_weights[i, j], [i, j]))
    error_weights_lst.sort(reverse=True)

    cmap_edges = []

    qubits = {i:None for i in range(n_qubits)}
    for curr_idx, edge in enumerate(error_weights_lst):
        # All qubits on the edge are used, skip
        if sum(i in qubits for i in edge[1]) == 0:
            continue
        # Edge may be added, choice of qubit to remove
        else:
            # Find the unused qubits
            unused_qubits = {i:0 for i in edge[1] if i in qubits}

            # Find the one with the next lowest edge
            n_found = 0
            for i in error_weights_lst[curr_idx + 1:]:
                # All found, break early
                if n_found == len(unused_qubits):
                    break
                # Check if the qubit is participating in this edge
                for j in unused_qubits:
                    if j in i[1] and unused_qubits[j] == 0:
                        unused_qubits[j] = i[0]
                        n_found += 1
            targ_qubit = min(unused_qubits)
            qubits.pop(targ_qubit)
            cmap_edges.append(edge[1])

    return cmap_edges

def build_err_profile_group(cal_matrices):
    
    err_types = ['single', 'paired']
    err_profile = {i:{} for i in err_types}
    counts = {}
    
    # Sum
    for t in err_types:
        for i in cal_matrices:
            for j in cal_matrices[i][t]:
                if j in err_profile[t]:
                    err_profile[t][j] += cal_matrices[i][t][j]
                    counts[j] += 1
                else:
                    err_profile[t][j] = cal_matrices[i][t][j]
                    counts[j] = 1
    # Normalise
    for t in err_types:
        for i in err_profile[t]:
            err_profile[t][i] /= counts[i]
    return err_profile