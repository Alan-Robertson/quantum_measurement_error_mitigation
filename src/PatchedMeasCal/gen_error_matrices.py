import numpy as np
import random
import copy
import qutip

from PatchedMeasCal.utils import f_dims, normalise
from functools import partial

class FakeMeasurementError():

    def __init__(self, error_arr_c, error_arr_u, error_arr_d, n_qubits = 5, s_penalty=0.3, coupling_map=None, meas_filter=None):
        self.probs = self.gen_probs(error_arr_c, error_arr_u, error_arr_d, n_qubits=n_qubits, s_penalty=0.3, coupling_map=coupling_map)
        self.n_qubits=n_qubits
        self.coupling_map=coupling_map
        self.meas_filter=meas_filter
        
    def __call__(self, counts, *args, **kwargs):
        return self.noisy_measure(counts, *args, **kwargs)

    def noisy_measure(self, counts):
        n_shots = sum(counts.values())

        # Vectorise counts
        vec = np.zeros((2 ** self.n_qubits, 1))
        for i in range(2 ** self.n_qubits):
            try:
                vec[i][0] = counts[str(bin(i)[2:].zfill(self.n_qubits))]
            except:
                pass
        err_counts = list(map(round, list((self.probs @ vec).flatten())))    
        counts_final = {}
        for i in range(2 ** self.n_qubits):
            i_str = bin(i)[2:].zfill(self.n_qubits)
            counts_final[i_str] = err_counts[i] 

        counts_final = self.sample_distribution(counts_final, n_shots)

        if self.meas_filter is not None:
            counts_final = self.meas_filter(counts_final)

        return counts_final

    def sub_set(self, n_qubits, participating_qubits=None):
        if participating_qubits is None:
            participating_qubits = list(range(n_qubits))

        #subset = np.array(self.probs)[:2 ** n_qubits, :2 ** n_qubits]

        subset = qutip.Qobj(np.array(self.probs), dims=f_dims(self.n_qubits)).ptrace(sel=participating_qubits)
        subset = normalise(np.array(subset))
        subset = list(map(list, subset.real))
        
        subset_error_obj = copy.deepcopy(self)
        subset_error_obj.n_qubits = n_qubits
        subset_error_obj.probs = subset

        if self.meas_filter is not None:
            subset_error_obj.meas_filter = partial(self.meas_filter, participating_qubits=participating_qubits)

        return subset_error_obj

    @staticmethod
    def sample_distribution(population, n_shots):
        # There are much more efficient ways to do this
        list_split = []
        for element in population:
            list_split += [element] * (population[element])
            
        list_pop = [random.choice(list_split) for _ in range(n_shots)]
        
        new_population = {}
        for element in population:
            new_population[element] = list_pop.count(element)
            
        new_population = {i:new_population[i] for i in new_population if new_population[i] > 0}
        return new_population

    @staticmethod
    def gen_probs(error_arr_c, error_arr_u, error_arr_d, n_qubits = 5, s_penalty=0.3, coupling_map=None):
        probs = [[0] * (2 ** n_qubits) for _ in range(2 ** n_qubits)]
            
        if len(error_arr_c) != n_qubits + 1:
            raise Exception("Incorrect Error Array")
        
        if len(error_arr_u) != n_qubits + 1:
            raise Exception("Incorrect Error Array")
             
        if len(error_arr_d) != n_qubits + 1:
            raise Exception("Incorrect Error Array")
        
        for row in range(2 ** n_qubits):
            row_str = bin(row)[2:].zfill(n_qubits)


            for col in range(2 ** n_qubits):
                col_str = bin(col)[2:].zfill(n_qubits)

                diff_str = [i - j for i, j in zip(list(map(int, row_str)), list(map(int, col_str)))]

                if coupling_map is not None:
                    n_edges = 0
                    n_err = sum(np.abs(diff_str))
                    for i, v_i in enumerate(diff_str):
                        for j, v_j in  enumerate(diff_str):
                            if i != j and v_i != 0 and v_j != 0:
                                if [i, j] in coupling_map:
                                    n_edges += 1
                    if n_err - 1 > n_edges:
                        continue
                
                #probs[row][col] -= s_penalty * sum(1 if i == 1 else 0 for i in row_str)
                probs[row][col] += error_arr_u[sum(1 if i == -1 else 0 for i in diff_str)]
                probs[row][col] += error_arr_d[sum(1 if i == 1 else 0 for i in diff_str)]
                probs[row][col] += error_arr_c[n_qubits - sum(1 if i == 0 else 0 for i in diff_str)]
                
                probs[row][col] = max(0, probs[row][col])
                
        #Normalise rows, we can then do arbitrary scaling factors in the error arr
        for row, _ in enumerate(probs):
            np_row = np.array(probs[row])
            if sum(np_row) > 0:
                np_row = np_row / sum(np_row) 
            probs[row] = list(np_row)
        
        return probs

