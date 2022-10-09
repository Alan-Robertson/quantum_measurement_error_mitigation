import qiskit
import copy

def design_circuit(n_qubits, inv_arr, circuit=None):
    '''
        Flips bits according to the inv array
    '''
    if circuit is None:
        circuit = qiskit.QuantumCircuit(n_qubits, n_qubits)
    
    for i, element in enumerate(inv_arr):
        if int(element) == 1: # In case of strings rather than integers
            circuit.x(i)
    
    # Cheating, we're going to measure all of them, because there are no nice mappings in qiskit and I don't want to have to traverse the dag :(
    # TODO: Traverse the DAG and re-implement the correct final measurements
    circuit.measure(list(range(n_qubits)), list(range(n_qubits)))
    return circuit

def strip_measurement(circuit):
    '''
        Creates a new version of the circuit sans measurement
    '''
    stripped_circuit = copy.deepcopy(circuit)
    i = 0
    while i < len(stripped_circuit.data):
        if stripped_circuit[i].operation.name == 'measure':
            stripped_circuit.data.pop(i)
            i -= 1
        i += 1
    return stripped_circuit

# SIM for even numbers of measured qubits 
def sim(circuit,
        backend,
        n_qubits,
        probs=None,
        n_shots=1000, # Number of shots for each string
        equal_shot_distribution=False # n_shots to be distributed equally
        ):
    
    # Strip measurements
    circuit = strip_measurement(circuit)

    sim_strs = [
        [0] * n_qubits, 
        [1] * n_qubits,
        [0, 1] * (n_qubits // 2), 
        [1, 0] * (n_qubits // 2)
    ]

    # Fixing up odd numbers of qubits
    for sim_str in sim_strs:
        if len(sim_str) < n_qubits:
            sim_str.append(sim_str[0])

    if equal_shot_distribution:
        n_shots //= len(sim_strs)
    
    # Inv and run    
    results = invert_build_run(circuit, n_qubits, backend, sim_strs, n_shots=n_shots, probs=probs)
    
    sim_results = additive_join_results(results)

    return sim_results


# AIM
def aim(circuit, # Circuit should not include measurement operators!
        backend,
        n_qubits, 
        probs=None, # Simulated error channel
        n_shots=1000,
        confirmation_shots = 1000,
        k=4, # Number of top results to use
        equal_shot_distribution = False,
        small_aim=False
       ):

    # Strip measurements
    eval_circuit = strip_measurement(circuit)

    # Build AIM Strs
    aim_strs = [[0] * n_qubits]
    if small_aim:  
        aim_strs = [[0] * n_qubits for _ in range(n_qubits)]
        for i in range(len(aim_strs)):
            aim_strs[i][i] = 1
    else:
        masks = [[1,0,0,1], [0,1,0,1], [1,0,1,0], [1,1,1,1]]
        for mask in masks:
            aim_strs += [([0] * (i * 2) + mask + [0] * (n_qubits - len(mask) - i * 2))[:n_qubits] for i in range((n_qubits - 1) // 2)]

    if equal_shot_distribution:
        confirmation_shots = int(0.75 * n_shots / k) # 3/4 of all shots on confirmation
        model_shots = int(0.25 * n_shots / len(aim_strs)) # 1/4 of all shots on model building
    else:
        confirmation_shots = n_shots
        model_shots = n_shots
    
    # Inv and run    
    model_results = invert_build_run(eval_circuit, n_qubits, backend, aim_strs, n_shots=model_shots, probs=probs)
    joint_results = additive_join_results(model_results)

    # Select top k strings
    string_strength = [(max(i.values()), arr) for i, arr in zip(model_results, aim_strs)]
    string_strength.sort(reverse=True)
    top_k = [arr for val, arr in string_strength[:k]]

    aim_results = invert_build_run(eval_circuit, n_qubits, backend, top_k, n_shots=confirmation_shots, probs=probs)
    aim_results = additive_join_results(aim_results)

    return aim_results


def invert_build_run(circuit, n_qubits, backend, inv_arrs, n_shots=1000, probs=None):
    tmp_circuits = []
    for inversion_arr in inv_arrs:

        tmp_circuit = copy.deepcopy(circuit)
        tmp_circuits.append(design_circuit(n_qubits, inversion_arr, circuit=tmp_circuit))

    job = qiskit.execute(tmp_circuits, backend, shots=n_shots)

    inv_results = job.result().get_counts()
    if probs is not None:
        for i, result in enumerate(results):
                noisy_measurement = measurement_error(result, n_qubits=n_qubits, probs=probs)
                results[i] = sample_distribution(noisy_measurement, shots)

    results = []
    for inv_result, arr in zip(inv_results, inv_arrs):
        # Invert strings
        result = {}
        for inv_meas_str in inv_result:
            meas_str = ''.join(
                map(
                    str, 
                    [i ^ j for i, j in zip(arr, map(int, list(inv_meas_str[::-1])))]
                    )
                )[::-1]

            result[meas_str] = inv_result[inv_meas_str]
        results.append(result)
    return results
            
def additive_join_results(results):
    # Join additively
    joint_results = {}
    for result in results:
        for meas_str in result:
            if meas_str in joint_results:
                joint_results[meas_str] += result[meas_str]
            else:
                joint_results[meas_str] = result[meas_str]
    return joint_results