import random
import math
import copy

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
from PatchedMeasCal.utils import vProgressbar, vprint, vtick

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

        self._coupling_graph = None

        self._edge_matrices = None
        self._patch_matrices = None
        self._meas_fitter = None

        self.n_qubits = len(self.backend.properties().qubits)
        self.n_shots = n_shots

        self.built = False # If not built, then need to build
        

    @property
    def cal_matrices(self):
        """
            Return cal_matrices
        """
        return self._meas_fitter

    def edge_to_int(self, edge):
        # Converts coupling map tuples to integers
        return edge[0] * self.n_qubits + edge[1]
  
    def build(self, probs=None, n_shots=None, verbose=False):
        '''
        Build a composite map fitter given a backend and a coupling map

        :: probs    :: Simulated error channel to apply
        :: n_shots  :: Total number of shots to take
        :: n_qubits :: Number of qubits to measure

        Returns a composite map calibration filter
        TODO Setup for build once, rebuild meas_fit for circs
        '''
        if n_shots is None:
            n_shots = self.n_shots

        vprint(verbose, "Building Coupling Graph")
        self._coupling_graph = CouplingMapGraph(self.coupling_map)
        self._coupling_map = self._coupling_graph.coupling_map

        # Create collection of fitters from mit_patterns
        vprint(verbose, "Building Edge Calibrations")
        self._edge_matrices = self.build_edge_calibrations(n_shots=n_shots, probs=probs, verbose=verbose)

        # Participating qubits will have already been factored into the previous graph construction
        vprint(verbose, "Building Patch Calibrations")
        self._patch_matrices = self.build_patch_calibrations(verbose=verbose)

        vprint(verbose, "Building Measure Fitter")
        self._meas_fitter = self.build_meas_fitter(verbose=verbose)

    def build_edge_calibrations(self, n_shots, verbose=True, probs=None):
        '''
            Builds calibrations for each edge via patching
        '''
        # Approximate two qubit error channel for calibration
        # These will also need to be sparse
        if probs is not None:
                pair_probs = probs.sub_set(2)

        # Fix our qubit layout
        # We will only be using adjacent edges so all operations should be legal
        initial_layout = list(range(self.n_qubits))

        # Construct calibration circuits
        # Execute calibration circuits independently
        # Simultaneous execution may exceed the maximum number of jobs, should implement a local job manager
        # at some point
        vprint(verbose, "\tBuilding Calibration Circuits")
        patches = self._coupling_graph.edge_patches(verbose=verbose)
        calibration_circs = self.construct_calibration_circuits(patches, verbose=verbose)
        transpiled_circs = qiskit.transpile(
            calibration_circs, 
            backend=self.backend, 
            initial_layout=initial_layout, 
            optimization_level=0
            ) 

        vprint(verbose, "\tExecuting Calibration Circuits")
        calibration_results = qiskit.execute(
            transpiled_circs,
            self.backend,
            shots=n_shots,
            initial_layout=initial_layout
            ).result()

        # Group calibration results in sets of four 
        # This will need to be modified for larger patch sizes, but for now we assume d=2
        calibration_counts = []
        vprint(verbose, "\tDe-hexing Measurement Results")
        for patch, patch_result in zip(patches, zip(*[iter(calibration_results.results)] * 4)):
            patch_results = []
            for result in patch_result:
                counts = result.data.counts

                bin_counts = {}
                # From hex to a binary string
                for res in counts:
                    key = bin(int(res[2:], 16))[2:].zfill(2 * len(patch))
                    bin_counts[key] = counts[res]

                # Apply fake probs
                if probs is not None:
                    bin_counts = pair_probs(bin_counts)

                patch_results.append(bin_counts)
            calibration_counts.append(patch_results)

        calibration_matrices = self.partial_calibration(patches, calibration_counts, n_shots=n_shots, verbose=verbose)

        edge_matrices = []
        for patch, matrices in zip(patches, calibration_matrices):
            for edge, edge_matrix in zip(patch, matrices):
                edge_matrices.append(CalibrationMatrix(edge_matrix, edge=edge))

        return edge_matrices


    def build_patch_calibrations(self, participating_qubits=None, verbose=False):
        # TODO: Participating qubits
        # We do this sparsely to account for hypothetical patches of larger size
        patch_matrices = copy.deepcopy(self._edge_matrices) 

        pb = vProgressbar(verbose, 20, self.n_qubits, "\tMerging Patches")
        for qubit in range(self.n_qubits):
            vtick(verbose, pb)
            num_participants = self._coupling_graph[qubit].n_edges

            # TODO: Modify for general construction beyond pairs
            participant_num = 0 # Order of construction
            for index, patch_matrix in enumerate(patch_matrices):
                # If this calibration matrix contains our target qubit
                if qubit in patch_matrix:
                    # Get the index of the qubit within the edge
                    position = patch_matrix.edge.index(qubit)

                    # Create sparse representation as a coo matrix within Qutip
                    cal_matrix = qutip.Qobj(
                        scipy.sparse.coo_matrix(
                            patch_matrix.calibration_matrix
                            ),
                        dims=self.f_dims(patch_matrix.n_edge_qubits)
                        )

                    # Take the partial trace for the single qubit approximation
                    single_qubit_approx = self.normalise(
                        np.array(
                            cal_matrix.ptrace(position)
                        )
                    )

                    # Construct left and right approximations
                    mean_approx_l = scipy.linalg.fractional_matrix_power(
                        single_qubit_approx,
                        (num_participants - 1 - participant_num) / num_participants
                    ).real
                    mean_approx_r = scipy.linalg.fractional_matrix_power(
                        single_qubit_approx,
                        participant_num / num_participants
                    ).real

                    # Expand and set to the correct terms, currently in order [a, b, I, I, I ...]
                    sparse_eye = scipy.sparse.eye(2 ** (patch_matrix.n_edge_qubits - 1))
                    expanded_approx_l = scipy.sparse.coo_matrix(mean_approx_l).tocsr()
                    expanded_approx_l = scipy.sparse.kron(expanded_approx_l, sparse_eye)
                    expanded_approx_l = qutip.Qobj(expanded_approx_l, dims=self.f_dims(patch_matrix.n_edge_qubits))

                    expanded_approx_r = scipy.sparse.coo_matrix(mean_approx_r).tocsr()
                    expanded_approx_r = scipy.sparse.kron(expanded_approx_r, sparse_eye)
                    expanded_approx_r = qutip.Qobj(expanded_approx_r, dims=self.f_dims(patch_matrix.n_edge_qubits))

                    # Construct permutation order
                    order = list(range(patch_matrix.n_edge_qubits))
                    order[position] = 0
                    order[0] = position

                    # Permute to correct order and retrieve sparse matrices
                    # CSC format is better for taking inv in the next step
                    expanded_approx_l = expanded_approx_l.permute(order)._data.tocsc()
                    expanded_approx_r = expanded_approx_r.permute(order)._data.tocsc()

                    # Invert
                    expanded_approx_l = scipy.sparse.linalg.inv(expanded_approx_l).real
                    expanded_approx_r = scipy.sparse.linalg.inv(expanded_approx_r).real

                    # Sparse rep of the calibration matrix
                    sparse_cal_matrix = scipy.sparse.csc_matrix(patch_matrix.calibration_matrix)

                    # All three objects are sparse, so the output is sparse
                    # This saves time later
                    patch_matrices[index] = CalibrationMatrix(
                            scipy.sparse.csc_matrix(
                                    expanded_approx_l @ sparse_cal_matrix @ expanded_approx_r
                                ),
                            edge=patch_matrix.edge
                            )
                    participant_num += 1
        return patch_matrices

    def construct_calibration_circuits(self, patches, verbose=False):
        '''
            Builds calibration circuits from patches
        '''
        circs = []
        pb = vProgressbar(verbose, 20, len(patches), "\tConstructing Calibration Circuits")
        for patch in patches:
            vtick(verbose, pb)
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

    def build_meas_fitter(self, participating_qubits=None, verbose=False):
        # Join local patches into a sparse calibration matrix
        # TODO non-complete sets of qubits
        if participating_qubits is not None:
            return self.build_partial_meas_fitter(participating_qubits=participating_qubits, verbose=verbose)
        
        meas_fitter = []
        # Reverse order of patch matrices
        pb = vProgressbar(verbose, 20, len(self._patch_matrices), "\tBuilding Meas Fitters from Patches")
        for patch_matrix in self._patch_matrices[::-1]:
            vtick(verbose, pb)
            
            # Invert each patch matrix
            inv_matrix = scipy.sparse.linalg.inv(patch_matrix.calibration_matrix).real

            # Convert to sparse qutip object 
            expanded_approx = scipy.sparse.kron(
                inv_matrix,
                scipy.sparse.eye(
                    2 ** (self.n_qubits - patch_matrix.n_edge_qubits),
                    dtype=np.float64)
                )
            expanded_approx = qutip.Qobj(expanded_approx, dims=self.f_dims(self.n_qubits))

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
            expanded_approx = expanded_approx.permute(order)._data.tocsc().real
            meas_fitter.append(CalibrationMatrix(expanded_approx, edge=patch_matrix.edge))
        return meas_fitter

    def build_partial_meas_fitter(self, participating_qubits=None, verbose=False):
        # Join local patches into a sparse calibration matrix
        # TODO non-complete sets of qubits
        
        n_qubits = len(participating_qubits)

        meas_fitter = []
        # Reverse order of patch matrices
        pb = vProgressbar(verbose, 20, len(self._patch_matrices), "\tBuilding Meas Fitters from Patches")
        for patch_matrix in self._patch_matrices[::-1]:
            vtick(verbose, pb)
            
            n_participating = sum([i in patch_matrix.edge for i in participating_qubits])
            if n_participating == 0:
                # Nothing contributes
                continue
            if n_participating < patch_matrix.n_edge_qubits:
                # Trace out over edges
                p_matrix = qutip.Qobj(
                                patch_matrix.calibration_matrix, 
                                dims=self.f_dims(patch_matrix.n_edge_qubits)
                            )
                partial_tr = [i for i, j in enumerate(patch_matrix.edge) if j in participating_qubits]
                p_matrix = self.normalise(
                        np.array(
                            p_matrix.ptrace(partial_tr)
                        )
                    )
                sp_matrix = scipy.sparse.csc_matrix(p_matrix)
                inv_matrix = scipy.sparse.linalg.inv(sp_matrix)

            else: 
                # Invert each patch matrix
                inv_matrix = scipy.sparse.linalg.inv(patch_matrix.calibration_matrix).real

            # Convert to sparse qutip object 
            expanded_approx = scipy.sparse.kron(
                inv_matrix,
                scipy.sparse.eye(
                    2 ** (n_qubits - n_participating),
                    dtype=np.float64)
                )
            expanded_approx = qutip.Qobj(expanded_approx, dims=self.f_dims(n_qubits))

            # Construct ordering for permutation
            # First n elements of the expanded approximation are non-identity and need to be correctly swapped
            # Last k elements are all the identity and may be freely interchanged
            order = []
            order_count = n_participating
            pair_count = 0
            for i in participating_qubits:
                if i in patch_matrix:
                    order.append(pair_count)
                    pair_count += 1
                else:
                    order.append(order_count)
                    order_count += 1

            # Permute the order of the tensor
            expanded_approx = expanded_approx.permute(order)._data.tocsc().real
            meas_fitter.append(CalibrationMatrix(expanded_approx, edge=patch_matrix.edge))
        return meas_fitter

    @staticmethod
    def partial_calibration(patches, calibration_results, n_shots=None, verbose=False):
        '''
            Reconstruct individual calibration matricies from the joint patch matrices
        '''
        if n_shots is None:
            n_shots = self.n_shots

        cal_matrices = []
        pb = vProgressbar(verbose, 20, len(patches), "\tTracing Patched Calibration Results")
        for patch, calibration_result in zip(patches, calibration_results):
            vtick(verbose, pb)
            # Construct calibration matrix for each edge in the patch
            patch_calibration_matrices = []

            # Construct calibration matrix for each edge in the patch
            for i, edge in enumerate(patch):
                cal_matrix = np.zeros((4, 4), dtype=np.float64)

                # Calibration matrix for one edge
                # i represents 00, 01, 10 and 11 circuits
                for j, result in enumerate(calibration_result):
                    cal_vec = np.zeros(4, dtype=np.float64)

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

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, measurement_results, participating_qubits=None, verbose=False):
        if participating_qubits is not None:
            meas_fitter = self.build_meas_fitter(participating_qubits=participating_qubits, verbose=verbose)
            n_qubits = len(participating_qubits)
        else:
            meas_fitter = self._meas_fitter
            n_qubits = self.n_qubits

        # These should be no more than the number of shots in size
        # Assumes correct mapping of qubits to vals
        rows = list(map(lambda x: int(x[::-1], 2), measurement_results.keys()))
        cols = [0] * len(measurement_results)
        values = list(measurement_results.values())
        n_shots = sum(values)

        results_vec = scipy.sparse.csc_array((values, (rows, cols)), shape=(2 ** n_qubits, 1))

        pb = vProgressbar(verbose, 20, len(self._meas_fitter), "Applying Meas Fitters")
        for meas_fit in meas_fitter[::-1]:
            vtick(verbose, pb)
            results_vec = meas_fit.calibration_matrix @ results_vec

            # Linear overhead to reduce matrix multiplication complexity
            # Negative values introduced by pseudo-inverse
            results_vec[results_vec < 0] = 0

        results_vec /= np.sum(results_vec)
        results_vec *= n_shots
        shot_results = {}
        for i, res in zip(results_vec.indices, results_vec.data):
            string = bin(i)[2:].zfill(n_qubits)[::-1] # To get back to qiskit's insane reversed strings
            shot_results[string] = res
        return shot_results

    def apply_meas_fitter(self, measurement_results, verbose=True):
        state_labels = self.state_labels(self.n_qubits)
        m_fitters = [CompleteMeasFitter(None, state_labels) for _ in self.meas_fit]

        # Convert back to dictionary

    @staticmethod
    def state_labels(n_qubits):
        return list(map(lambda x: bin(x)[2:], range(2 ** n_qubits)))

    @staticmethod
    def normalise(x):
        '''
            Normalise the partial trace of a calibration matrix
        '''
        for i in range(x.shape[1]):
            tot = sum(x[:, i])
            if tot != 0:
                x[:, i] /= tot
        return x

    @staticmethod
    def f_dims(n):
        '''
            Dimension ordering for n qubits
        '''
        return [[2 for i in range(n)]] * 2







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
    