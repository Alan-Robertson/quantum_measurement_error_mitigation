import random
import math

from typing import List, Union
from functools import reduce, partial

import numpy as np
from qutip import *
import scipy

import qiskit
from qiskit import IBMQ
from qiskit import transpile, QuantumRegister, assemble
from qiskit import QuantumCircuit, execute, Aer, QuantumCircuit
from qiskit.ignis.mitigation.measurement import complete_meas_cal, tensored_meas_cal, CompleteMeasFitter, TensoredMeasFitter

from edge_bfs import edge_patches

class CalibrationMatrix():
    def __init__(self, calibration_matrix, edge=None):
        self.calibration_matrix = calibration_matrix
        self.edge = edge

class TensorPatchFitter():
    """
    Build a composite map fitter given a backend and a coupling map
    Uncertain why this and the other associated objects don't inherit from some class.

        :: backend  :: Backend object with associated coupling map
        :: coupling_map :: Coupling map if backend does not have one
    """

    def __init__(self,
        backend,
        coupling_map=None):

        self.backend = backend
        # Coupling map override - if custom coupling map is provided
        if coupling_map is None:
            coupling_map = self.backend.configuration().coupling_map
        else:
            self.coupling_map = coupling_map

        self._edge_matrices = None
        self._patch_matrices = None
        self._coupling_graph = None

        self.n_qubits = len(self.backend.properties().qubits)
        

    @property
    def cal_matrices(self):
        """
            Return cal_matrices
        """
        return self._cal_matrices

    def edge_to_int(self, edge)
        # Converts coupling map tuples to integers
        return edge[0] * self.n_qubits + edge[1]
  


    def build(self, probs=None, n_shots=1000, assert_accuracy=False, shots_fixed=True):
        '''
        Build a composite map fitter given a backend and a coupling map


        :: probs    :: Simulated error channel to apply
        :: n_shots  :: Total number of shots to take
        :: shots_fixed :: Constant number of shots, default is for comparing methods for equal numbers of shots
        :: n_qubits :: Number of qubits to measure
        :: assert_accuracy :: Check accuracy of approximation

        Returns a composite map calibration filter
        '''

# Unique Assignment and clear duplicates
# Maps edges to integers
mit_patterns = {}
for pat in coupling_map:
    # Cull couplings with qubits that aren't in our register
    out_of_scope = False
    for i in pat:
        if i >= n_qubits:
            out_of_scope = True

    # Cull duplicate couplings
    if not out_of_scope:
        if cmap_to_mit(*pat) not in mit_patterns and cmap_to_mit(*pat[::-1]) not in mit_patterns:
            mit_patterns[self.edge_to_int(*pat)] = pat

    # Number of shots per trial
    if shots_fixed:
        shots = n_shots
    else:
        shots = n_shots // (len(mit_patterns) * 4)

    self._coupling_graph = CouplingMapGraph(self.coupling_map)

    # Create collection of fitters from mit_patterns
    self._edge_matrices = self.construct_edge_calibrations(shots, probs=probs)

    # Participating qubits will have already been factored into the previous graph construction
    self._patch_matrices = self.construct_patch_calibrations()



def construct_patch_calibrations(self, participating_qubits):

    # TODO: Fix up the participating qubits bit
    coupling_graph = CouplingMapGraph(self.coupling_map, participating_qubits=participating_qubits)

    patch_matrices = copy.deepcopy(self._cal_matrices)

    for qubit in range(self.n_qubits):
        num_participants = coupling_graph.graph[i].num_edges

        # Modify for general construction beyond pairs
        participant_num = 0 # Order of construction
        for index in patch_matrices:
            if qubit in patch_matrices[index].edge:

                # Get the index of the qubit within the edge
                position = patch_matrices[index].edge.index(qubit)

                # Create sparse representation as a coo matrix within Qutip
                qcal_matrix = Qobj(
                    scipy.sparse.coo_matrix(
                        patch_matrices[index].calibration_matrix
                        )
                    )

                # Take the partial trace for the single qubit approximation
                single_qubit_approx = normalise(
                    np.array(
                        qcal_matrix.ptrace(position)
                    )
                )

                # Construct left and right approximations
                mean_approx_l = scipy.linalg.fractional_matrix_power(
                    single_qubit_approx,
                    (num_participants - 1 - participant_num) / num_participants
                )
                mean_approx_r = scipy.linalg.fractional_matrix_power(
                    single_qubit_approx,
                    participant_num / num_participants
                )

                # Expand and set to the correct terms, currently in order [a, b, I, I, I ...]
                sparse_eye = scipy.sparse.eye(2 ** (len(mit_patterns[pat]) - 1))
                expanded_approx_l = scipy.sparse.coo_matrix(mean_approx_l).tocsr()
                expanded_approx_l = scipy.sparse.kron(expanded_approx_l, sparse_eye)
                expanded_approx_l = Qobj(expanded_approx_l, dims=f_dims(len(mit_patterns[pat])))

                expanded_approx_r = scipy.sparse.coo_matrix(mean_approx_r).tocsr()
                expanded_approx_r = scipy.sparse.kron(expanded_approx_r, sparse_eye)
                expanded_approx_r = Qobj(expanded_approx_r, dims=f_dims(len(mit_patterns[pat])))

                # Construct permutation order
                order = list(range(len(mit_patterns[pat])))
                order[position] = 0
                order[0] = position

                # Permute to correct order and retrieve sparse matrices
                # CSC format is better for taking inv in the next step
                expanded_approx_l = expanded_approx_l.permute(order)._data.tocsc()
                expanded_approx_r = expanded_approx_r.permute(order)._data.tocsc()

                # Invert
                expanded_approx_l = scipy.sparse.linalg.inv(expanded_approx_l)
                expanded_approx_r = scipy.sparse.linalg.inv(expanded_approx_r)            

                # All three objects are sparse, so the output is sparse
                patch_matrices[index] = (
                      expanded_approx_l 
                    @ qubit_pair_fitters[pat] 
                    @ expanded_approx_r
                )
                participant_num += 1
    return patch_matrices
                







#######################


        # Join calibration matrices into local patches
        for i in range(self.n_qubits):

            # Count number of participating matricies
            num_participants = 0
            for j in mit_patterns:
                if i in mit_patterns[j]:
                    num_participants += 1

            # Modify for general construction beyond pairs
            participant_num = 0 # Order of construction
            for pat in mit_patterns:
                if i in mit_patterns[pat]:

                    position = mit_patterns[pat].index(i)

                    cal_matrix = Qobj(qubit_pair_fitters[pat], dims=f_dims(2))

                    # Ptrace to approximate the target qubit
                    single_qubit_approx = normalise(np.array(cal_matrix.ptrace(position)))

                    # Construct left and right approximations
                    mean_approx_l = scipy.linalg.fractional_matrix_power(
                        single_qubit_approx,
                        (num_participants - 1 - participant_num) / num_participants
                    )

                    mean_approx_r = scipy.linalg.fractional_matrix_power(
                        single_qubit_approx,
                        participant_num / num_participants
                    )
                
                    # Check accuracy of approximation
                    if assert_accuracy:
                        assert(np.linalg.norm(
                             mean_approx_l 
                           @ scipy.linalg.fractional_matrix_power(single_qubit_approx, 1 / num_participants)
                           @ mean_approx_r
                           - single_qubit_approx
                        ) < 1e-4)

                    # Expand and set to the correct terms, currently in order [a, b, I, I, I ...]
                    sparse_eye = scipy.sparse.eye(2 ** (len(mit_patterns[pat]) - 1))
                    expanded_approx_l = scipy.sparse.coo_matrix(mean_approx_l).tocsr()
                    expanded_approx_l = scipy.sparse.kron(expanded_approx_l, sparse_eye)
                    expanded_approx_l = Qobj(expanded_approx_l, dims=f_dims(len(mit_patterns[pat])))

                    expanded_approx_r = scipy.sparse.coo_matrix(mean_approx_r).tocsr()
                    expanded_approx_r = scipy.sparse.kron(expanded_approx_r, sparse_eye)
                    expanded_approx_r = Qobj(expanded_approx_r, dims=f_dims(len(mit_patterns[pat])))

                    # Construct permutation order
                    order = list(range(len(mit_patterns[pat])))
                    order[position] = 0
                    order[0] = position

                    # Permute to correct order and retrieve sparse matrices
                    # CSC format is better for taking inv in the next step
                    expanded_approx_l = expanded_approx_l.permute(order)._data.tocsc()
                    expanded_approx_r = expanded_approx_r.permute(order)._data.tocsc()

                    # Invert
                    expanded_approx_l = scipy.sparse.linalg.inv(expanded_approx_l)
                    expanded_approx_r = scipy.sparse.linalg.inv(expanded_approx_r)            

                    # All three objects are sparse, so the output is sparse
                    qubit_pair_fitters[pat] = (
                          expanded_approx_l 
                        @ qubit_pair_fitters[pat] 
                        @ expanded_approx_r
                    )
                    participant_num += 1

        # Join local patches into a full calibration matrix
        for i in range(self.n_qubits):
            for pat in mit_patterns:

                pair = mit_patterns[pat]
                pair_approx = qubit_pair_fitters[pat]

                expanded_approx = scipy.sparse.kron(pair_approx, scipy.sparse.eye(2 ** (n_qubits - len(pair))))
                expanded_approx = Qobj(expanded_approx, dims=f_dims(self.n_qubits))

                # Construct ordering for permutation
                # First n elements of the expanded approximation are non-identity and need to be correctly swapped
                # Last k elements are all the identity and may be freely interchanged
                order = []
                order_count = len(pair)
                pair_count = 0
                for i in range(self.n_qubits):
                    if i in pair:
                        order.append(pair_count)
                        pair_count += 1
                    else:
                        order.append(order_count)
                        order_count += 1

                # Apply permutation
                expanded_approx = expanded_approx.permute(order)._data.tocsc()
                qubit_pair_fitters[pat] = expanded_approx

                self._cal_matrices = qubit_pair_fitters

        # Probably can crop here
        # Base calibration matrix
        cal_matrix = np.eye(2 ** self.n_qubits)
        for pat in mit_patterns:
            cal_matrix = qubit_pair_fitters[pat] @ cal_matrix
        
        cal_matrix = np.real(cal_matrix)
            
        # Build new cal matrix object:
        state_labels = [str(bin(i)[2:]).zfill(self.n_qubits) for i in range(2 ** self.n_qubits)]
        fitter = CompleteMeasFitter(results=None, state_labels=state_labels)
        
        # Set the corresponding objects appropriately
        fitter._tens_fitt.cal_matrices = [cal_matrix]
        return fitter

    def construct_edge_calibrations(self, shots, probs=None):
        # Approximate two qubit error channel for calibration
        # These will also need to be sparse
        if probs is not None:
                pair_probs = np.array(probs)[:4, :4]

        # Build a register for calibrations
        qr = qiskit.QuantumRegister(n_qubits)

        # Fix our qubit layout
        # We will only be using adjacent edges so all operations should be legal
        initial_layout = list(range(n_qubits))

        # Construct calibration circuits
        # Execute calibration circuits independently
        # Simultaneous execution may exceed the maximum number of jobs, should implement a local job manager
        # at some point
        patches = edge_patches(coupling_map)
        calib_circuits = construct_calib_circuits(patches)
        transpiled_circs = map(partial(transpile, initial_layout=initial_layout, optimization_level=0), calib_circuits) 

        calibration_results = [execute(i, backend, shots=shots, initial_layout=initial_layout).result() for i in transpiled_circs]

        calibration_counts = []
        for patch, calibration_result in zip(patches, calibration_results):
            patch_results = []
            for result in calibration_result.results:
                counts = result.data.counts
                bin_counts = {}
                # From hex to a binary string
                for res in counts:
                    key = bin(int(res[2:], 16))[2:].zfill(2 * len(patch))
                    bin_counts[key] = counts[res]
                patch_results.append(bin_counts)
            calibration_counts.append(patch_results)

        calibration_matrices = edge_calibration(patches, calibration_counts, shots)
        calibration_matrices = reduce(lambda x, y: x + y, calibration_matrices)
        edges = reduce(lambda x, y: x + y, edges)
        
        calibration_data = {}
        for cal_matrix, edge in zip(calibration_matrices, edges):
            index = edge_to_int(edge)
        calibration_data[index] = CalibrationMatrix(cal_matrix, edge=edge)
        
        return calibration_data

def partial_calibration(patches, calibration_results, shots):
    '''
        Reconstruct individual calibration matricies from the joint matrices
    '''
    cal_matrices = []
    for patch, calibration_result in zip(patches, calibration_results):
        # Construct calibration matrix for each edge in the patch
        patch_calibration_matrices = []

        # Construct calibration matrix for each edge in the patch
        for i, edge in enumerate(patch):
            cal_matrix = np.zeros((4, 4), dtype=np.float32)

            # Calibration matrix for one edge
            # i represents 00, 01, 10 and 11 circuits
            for j, result in enumerate(calibration_result):
                cal_vec = np.zeros(4, dtype=np.float32)

                # Partial trace over the rest of the distribution
                # There are probably speedups here
                for res in result:
                    cal_vec[int(res[2 * i] + res[2 * i + 1], 2)] += result[res]
                cal_matrix[j] = cal_vec / shots
                
            # We'll invert the calibration matrix later
            patch_calibration_matrices.append(cal_matrix)
        
        # Qiskit's result strings are reversed for reasons known only to them
        cal_matrices.append(patch_calibration_matrices[::-1])
    return cal_matrices


################

    def construct_calib_circuits(self, patches):
        circs = []
        for patch in patches:
            circuits = [QuantumCircuit(self.n_qubits, 2 * len(patch)) for _ in range(4)]

            # Four measurement calibration circuits: 00, 01, 10, 11
            for i in range(4):
                    if i == 1:
                        for edge in patch:
                            circuits[i].x(edge[0])
                    elif i == 2:
                        for edge in patch:
                            circuits[i].x(edge[1])
                    elif i == 3:
                        for edge in patch:
                            circuits[i].x(edge[0])
                            circuits[i].x(edge[1])
                    # Measure all relevant qubits
                    circuits[i].measure(reduce(lambda x, y: x + y, patch), range(2 * len(patch)))
            circs.append(circuits)
        return circs

################

def composite_filter(circuit, probs=None, n_shots=1000, n_qubits=4, **kwargs):
    '''
        composite_filter
        Performs all qubit measurement error calibration on a target circuit 
        Using coupling map pairs to build a composite filter
        
        :: circuit  :: Circuit to perform measurement error calibration over
        :: probs    :: Simulated error channel to apply
        :: n_shots  :: Number of shots to take
        :: n_qubits :: Number of qubits to measure
        
        Returns the shot statistics following calibration
    '''
    shots = n_shots # 50% of shots on building the filter
    confirmation_shots = n_shots 
    
    # Build Calibration Matrix
    qr = QuantumRegister(n_qubits)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    
    # Build the filter
    comp_filter = composite_map(backend, n_qubits=n_qubits, probs=probs, n_shots=shots, **kwargs)
    
    # Add measurements to circuit
    circuit = design_circuit(n_qubits, '0' * n_qubits, circuit=circuit)

    # Once the fitter is constructed, attempt the real experiments
    job = execute(circuit, backend, shots=confirmation_shots)
    result = job.result()
    
    if probs is not None:
        cal_res_measurement_error(result, probs, n_qubits=n_qubits)
        
    result = comp_filter.filter.apply(result).get_counts()
    
    # Results are in reverse order
    qiskit_results_qubit_order = {}
    for i in result:
        qiskit_results_qubit_order[i[::-1]] = result[i]
    return qiskit_results_qubit_order



def normalise(x):
    '''
        Normalise the partial trace of a calibration matrix
    '''
    for i in range(x.shape[1]):
        tot = sum(x[:, i])
        if tot != 0:
            x[:, i] /= tot
    return x

def f_dims(n):
    '''
        Dimension ordering for n qubits
    '''
    return [[2 for i in range(n)]] * 2


def cal_res_measurement_error(
    cal_results,
    probs : list,
    n_qubits=4
    ) -> dict:   
    '''
        cal_res_measurement_error
        Calculates the output after applying a simulated measurement error
        
        :: cal_results : results object :: Results before the measurement error
        :: probs       : list :: Measurement error to apply
        :: n_qubits    : int  :: The number of qubits to apply over
        
        Acts in place on the cal_results object
    '''
    
    # Loop over results and construct counts
    for i, res in enumerate(cal_results.results):
        d = {}
        cd = res.data.to_dict()['counts']
        for key in cd:
            d[bin(int(key, 16))[2:].zfill(n_qubits)] = cd[key]
        
        # Apply measurement errors to the counts
        counts = measurement_error(d, n_qubits=n_qubits, probs=probs)

        # Fix the keys back to hex formatting
        data_counts = {}
        for key in counts:
            data_counts[hex(int(key, 2))] = counts[key]
        cal_results.results[i].data.counts = data_counts
        
    return
    