import numpy as np
import random
import copy

class FakeMeasurementError():

    def __init__(self, error_arr_c, error_arr_u, error_arr_d, n_qubits = 5, s_penalty=0.3):
        self.probs = self.gen_probs(error_arr_c, error_arr_u, error_arr_d, n_qubits=n_qubits, s_penalty=0.3)
        self.n_qubits=n_qubits
        
    def __call__(self, counts):
        return self.noisy_measure(counts)

    def noisy_measure(self, counts):
        n_shots = sum(counts.values())

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

        return self.sample_distribution(counts_final, n_shots)

    def sub_set(self, n_qubits):
        subset = np.array(self.probs)[:2 ** n_qubits, :2 ** n_qubits]
        for i in range(2 ** n_qubits):
            norm_val = np.sum(subset[:, i])
            subset[:, i] /= norm_val
        subset_error_obj = copy.deepcopy(self)
        subset_error_obj.n_qubits = n_qubits
        subset_error_obj.probs = subset
        return subset_error_obj

    @staticmethod
    def sample_distribution(population, n_shots):
        # There are much more efficient ways to do this
        list_split = []
        for element in population:
            list_split += [element] * population[element]
            
        list_pop = [random.choice(list_split) for _ in range(n_shots)]
        
        new_population = {}
        for element in population:
            new_population[element] = list_pop.count(element)
            
        new_population = {i:new_population[i] for i in new_population if new_population[i] > 0}
        return new_population

    @staticmethod
    def gen_probs(error_arr_c, error_arr_u, error_arr_d, n_qubits = 5, s_penalty=0.3):
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