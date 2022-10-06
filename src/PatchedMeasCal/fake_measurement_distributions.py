import random
import numpy as np
import scipy


def sample_distribution(population, n_shots):
    # There are much more efficient ways to do this
    list_split = []
    for element in population:
        list_split += [element] * population[element]
        
    list_pop = [random.choice(list_split) for _ in range(n_shots)]
    
    new_population = {}
    for element in population:
        new_population[element] = list_pop.count(element)

    return new_population


def renormalise_measurement_results(measurement_results, norm_shots):
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

    # Filter negative values if there are any
    results_vec[results_vec < 0] = 0

    results_vec /= np.sum(results_vec)
    results_vec *= norm_shots

    shot_results = {}
    for i, res in zip(results_vec.indices, results_vec.data):
        string = bin(i)[2:].zfill(n_qubits)[::-1] # To get back to qiskit's insane reversed strings
        shot_results[string] = res
    return shot_results