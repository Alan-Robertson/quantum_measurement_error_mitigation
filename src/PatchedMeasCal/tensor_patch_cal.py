import random
import math

from typing import List, Union
from functools import reduce, partial

import numpy as np
import qutip
import scipy

import qiskit
from qiskit import IBMQ
from qiskit import transpile, QuantumRegister, assemble
from qiskit import QuantumCircuit, execute, Aer, QuantumCircuit
from qiskit.ignis.mitigation.measurement import complete_meas_cal, tensored_meas_cal, CompleteMeasFitter, TensoredMeasFitter

from PatchedMeasCal.edge_bfs import CouplingMapGraph
from PatchedMeasCal.utils import f_dims, normalise

class CalibrationMatrix():
    def __init__(self, calibration_matrix, edge=None):
        self.calibration_matrix = calibration_matrix
        self.edge = edge
        self.n_edge_qubits = len(self.edge)
    def __repr__(self):
        return str(self.edge) + str(self.calibration_matrix)
    def __str__(self):
        return self.__repr__()
    def __contains__(self, i):
        return self.edge.__contains__(i)
        

class TensorPatchFitter():
    """
    Build a composite map fitter given a backend and a coupling map
    Uncertain why this and the other associated objects don't inherit from some class.

        :: backend  :: Backend object with associated coupling map
        :: coupling_map :: Coupling map if backend does not have one
    """

    def __init__(self,
        backend,
        coupling_map=None,
        n_shots = 4000):

        self.backend = backend
        # Coupling map override - if custom coupling map is provided
        if coupling_map is None:
            self.coupling_map = self.backend.configuration().coupling_map
        else:
            self.coupling_map = coupling_map

        self._edge_matrices = None
        self._patch_matrices = None
        self._coupling_graph = None

        self.n_qubits = len(self.backend.properties().qubits)
        self.n_shots = n_shots

        self.built = False # If not built, then need to build
        

    @property
    def cal_matrices(self):
        """
            Return cal_matrices
        """
        return self._cal_matrices

    def edge_to_int(self, edge):
        # Converts coupling map tuples to integers
        return edge[0] * self.n_qubits + edge[1]
  
    def build(self, probs=None, n_shots=None, assert_accuracy=False, shots_fixed=True):
        '''
        Build a composite map fitter given a backend and a coupling map

        :: probs    :: Simulated error channel to apply
        :: n_shots  :: Total number of shots to take
        :: shots_fixed :: Constant number of shots, default is for comparing methods for equal numbers of shots
        :: n_qubits :: Number of qubits to measure
        :: assert_accuracy :: Check accuracy of approximation

        Returns a composite map calibration filter
        '''
        if n_shots is None:
            n_shots = self.n_shots

        self._coupling_graph = CouplingMapGraph(self.coupling_map)

        # Create collection of fitters from mit_patterns
        self._edge_matrices = self.construct_edge_calibrations(n_shots=n_shots, probs=probs)

        # Participating qubits will have already been factored into the previous graph construction
        self._patch_matrices = self.construct_patch_calibrations()


    def construct_patch_calibrations(self, participating_qubits=None):
        # TODO: Participating qubits
        #patches = self._coupling_graph.edge_patches(participating_qubits=participating_qubits)
        # We do this sparsely to account for hypothetical patches of larger size
        patch_matrices = [] 

        for qubit in range(self.n_qubits):
            num_participants = self._coupling_graph[qubit].n_edges

            # TODO: Modify for general construction beyond pairs
            participant_num = 0 # Order of construction
            for edge_matrix in self._edge_matrices:
                # If this calibration matrix contains our target qubit
                if qubit in edge_matrix:
                    # Get the index of the qubit within the edge
                    position = edge_matrix.edge.index(qubit)

                    # Create sparse representation as a coo matrix within Qutip
                    cal_matrix = qutip.Qobj(
                        scipy.sparse.coo_matrix(
                            edge_matrix.calibration_matrix
                            ),
                        dims=f_dims(edge_matrix.n_edge_qubits)
                        )

                    # Take the partial trace for the single qubit approximation
                    single_qubit_approx = normalise(
                        np.array(
                            cal_matrix.ptrace(position)
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
                    sparse_eye = scipy.sparse.eye(2 ** (edge_matrix.n_edge_qubits - 1))
                    expanded_approx_l = scipy.sparse.coo_matrix(mean_approx_l).tocsr()
                    expanded_approx_l = scipy.sparse.kron(expanded_approx_l, sparse_eye)
                    expanded_approx_l = qutip.Qobj(expanded_approx_l, dims=f_dims(edge_matrix.n_edge_qubits))

                    expanded_approx_r = scipy.sparse.coo_matrix(mean_approx_r).tocsr()
                    expanded_approx_r = scipy.sparse.kron(expanded_approx_r, sparse_eye)
                    expanded_approx_r = qutip.Qobj(expanded_approx_r, dims=f_dims(edge_matrix.n_edge_qubits))

                    # Construct permutation order
                    order = list(range(edge_matrix.n_edge_qubits))
                    order[position] = 0
                    order[0] = position

                    # Permute to correct order and retrieve sparse matrices
                    # CSC format is better for taking inv in the next step
                    expanded_approx_l = expanded_approx_l.permute(order)._data.tocsc()
                    expanded_approx_r = expanded_approx_r.permute(order)._data.tocsc()

                    # Invert
                    expanded_approx_l = scipy.sparse.linalg.inv(expanded_approx_l)
                    expanded_approx_r = scipy.sparse.linalg.inv(expanded_approx_r)

                    # Sparse rep of the calibration matrix
                    sparse_cal_matrix = scipy.sparse.csc_matrix(edge_matrix.calibration_matrix)

                    # All three objects are sparse, so the output is sparse
                    # This saves time later
                    patch_matrices.append(
                            CalibrationMatrix(
                              expanded_approx_l 
                            @ sparse_cal_matrix
                            @ expanded_approx_r,
                            edge=edge_matrix.edge
                            )
                        )
                    participant_num += 1

        return patch_matrices

    def construct_edge_calibrations(self, n_shots, probs=None):
        '''
            Builds calibrations for each edge via patching
        '''
        # Approximate two qubit error channel for calibration
        # These will also need to be sparse
        if probs is not None:
                pair_probs = np.array(probs)[:4, :4]

        # Build a register for calibrations
        qr = qiskit.QuantumRegister(self.n_qubits)

        # Fix our qubit layout
        # We will only be using adjacent edges so all operations should be legal
        initial_layout = list(range(self.n_qubits))

        # Construct calibration circuits
        # Execute calibration circuits independently
        # Simultaneous execution may exceed the maximum number of jobs, should implement a local job manager
        # at some point
        patches = self._coupling_graph.edge_patches()
        calibration_circs = self.construct_calibration_circuits(patches)
        transpiled_circs = qiskit.transpile(
            calibration_circs, 
            backend=self.backend, 
            initial_layout=initial_layout, 
            optimization_level=0) 

        calibration_results = execute(
            transpiled_circs,
            self.backend,
            shots=n_shots,
            initial_layout=initial_layout).result()

        # Group calibration results in sets of four 
        # This will need to be modified for larger patch sizes, but for now we assume d=2
        calibration_counts = []
        for patch, patch_result in zip(patches, zip(*[iter(calibration_results.results)] * 4)):
            patch_results = []
            for result in patch_result:
                counts = result.data.counts
                bin_counts = {}
                # From hex to a binary string
                for res in counts:
                    key = bin(int(res[2:], 16))[2:].zfill(2 * len(patch))
                    bin_counts[key] = counts[res]

                patch_results.append(bin_counts)
            calibration_counts.append(patch_results)

        calibration_matrices = self.partial_calibration(patches, calibration_counts, n_shots=n_shots)

        edge_matrices = []
        for patch, matrices in zip(patches, calibration_matrices):
            for edge, edge_matrix in zip(patch, matrices):
                edge_matrices.append(CalibrationMatrix(edge_matrix, edge=edge))

        return edge_matrices

    def construct_calibration_circuits(self, patches):
        '''
            Builds calibration circuits from patches
        '''
        circs = []
        for patch in patches:
            calibration_circuits = [qiskit.QuantumCircuit(self.n_qubits, 2 * len(patch)) for _ in range(4)]

            # Four measurement calibration circuits: 00, 01, 10, 11
            for i in range(4):
                    if i == 1:
                        for edge in patch:
                            calibration_circuits[i].x(edge[0])
                    elif i == 2:
                        for edge in patch:
                            calibration_circuits[i].x(edge[1])
                    elif i == 3:
                        for edge in patch:
                            calibration_circuits[i].x(edge[0])
                            calibration_circuits[i].x(edge[1])
                    # Measure all relevant qubits
                    calibration_circuits[i].measure(reduce(lambda x, y: x + y, patch), range(2 * len(patch)))
            circs += calibration_circuits
        return circs

    def build_meas_fitter(self, participating_qubits=None):
        # Join local patches into a sparse calibration matrix
        # TODO non-complete sets of qubits
        meas_fitter = []
        # Reverse order of patch matrices
        for patch_matrix in self._patch_matrices[::-1]:
            
            # Invert each patch matrix
            inv_matrix = scipy.sparse.linalg.inv(patch_matrix.calibration_matrix)

            # Convert to sparse qutip object 
            expanded_approx = scipy.sparse.kron(
                inv_matrix,
                scipy.sparse.eye(
                    2 ** (self.n_qubits - patch_matrix.n_edge_qubits
                        )
                    )
                )
            expanded_approx = qutip.Qobj(expanded_approx, dims=f_dims(self.n_qubits))

            # Construct ordering for permutation
            # First n elements of the expanded approximation are non-identity and need to be correctly swapped
            # Last k elements are all the identity and may be freely interchanged
            order = []
            order_count = patch_matrix.n_edge_qubits
            pair_count = 0
            for i in range(self.n_qubits):
                if i in patch_matrix:
                    order.append(pair_count)
                    pair_count += 1
                else:
                    order.append(order_count)
                    order_count += 1

            # Permute the order of the tensor
            expanded_approx = expanded_approx.permute(order)._data.tocsc()
            meas_fitter.append(CalibrationMatrix(expanded_approx, edge=patch_matrix.edge))
        return meas_fitter

    @staticmethod
    def partial_calibration(patches, calibration_results, n_shots=None):
        '''
            Reconstruct individual calibration matricies from the joint patch matrices
        '''
        if n_shots is None:
            n_shots = self.n_shots

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
                    cal_matrix[j] = cal_vec / n_shots
                    
                # We'll invert the calibration matrix later
                patch_calibration_matrices.append(cal_matrix)
            
            # Qiskit's result strings are reversed for reasons known only to them
            cal_matrices.append(patch_calibration_matrices[::-1])
        return cal_matrices

        def apply(self, measurement_results):
            pass


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
    qr = QuantumRegister(self.n_qubits)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    
    # Build the filter
    comp_filter = composite_map(backend, n_qubits=self.n_qubits, probs=probs, n_shots=shots, **kwargs)
    
    # Add measurements to circuit
    circuit = design_circuit(self.n_qubits, '0' * self.n_qubits, circuit=circuit)

    # Once the fitter is constructed, attempt the real experiments
    job = execute(circuit, backend, shots=confirmation_shots)
    result = job.result()
    
    if probs is not None:
        cal_res_measurement_error(result, probs, n_qubits=self.n_qubits)
        
    result = comp_filter.filter.apply(result).get_counts()
    
    # Results are in reverse order
    qiskit_results_qubit_order = {}
    for i in result:
        qiskit_results_qubit_order[i[::-1]] = result[i]
    return qiskit_results_qubit_order


    


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
    