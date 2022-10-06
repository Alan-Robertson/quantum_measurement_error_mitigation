import qiskit
import copy

def design_circuit(n_qubits, inv_arr, circuit=None):
    '''
        Flips bits according to the inv array
    '''
    if circuit is None:
        circuit = qiskit.QuantumCircuit(n_qubits, n_qubits)
    
    for i, element in enumerate(inv_arr):
        if int(element) == 1:
            circuit.x(i)
    
    circuit.measure(list(range(n_qubits)), list(range(n_qubits)))
    return circuit

# SIM for even numbers of measured qubits 
def sim(circuit,
        backend,
        n_qubits,
        probs=None,
        n_shots=1000, # Number of shots for each string
        equal_shot_distribution=False # n_shots to be distributed equally
        ):
    
    # Strip measurements
    circuit = copy.deepcopy(circuit)
    circuit.remove_final_measurements()

    sim_strs = [
        [0] * n_qubits, 
        [1] * n_qubits,
        [0, 1] * (n_qubits // 2), 
        [1, 0] * (n_qubits // 2)
    ]

    if equal_shot_distribution:
        n_shots //= len(sim_strs)
        
    sim_results = {}
    for inversion_arr in sim_strs:

        tmp_circuit = copy.deepcopy(circuit)
        tmp_circuit = design_circuit(n_qubits, inversion_arr, circuit=tmp_circuit)

        job = qiskit.execute(tmp_circuit, backend, shots=n_shots)

        results = job.result().get_counts()

        if probs is not None:
            noisy_measurement = measurement_error(results, n_qubits=n_qubits, probs=probs)
            results = sample_distribution(noisy_measurement, shots)

        for count in results:
            # Invert SIM strings
            count_arr = ''.join(map(str, [i ^ j for i, j in zip(inversion_arr, map(int, list(count[::-1])))]))

            if count_arr in sim_results:
                sim_results[count_arr] += results[count]
            else:
                sim_results[count_arr] = results[count]

    return sim_results


# AIM
def aim(circuit, # Circuit should not include measurement operators!
        backend,
        n_qubits, 
        probs=None, # Simulated error channel
        n_shots=1000,
        confirmation_shots = 1000,
        k=4, # Number of top results to use
        equal_shot_distribution = False
       ):
    
    # Strip measurements
    circuit = copy.deepcopy(circuit)
    circuit.remove_final_measurements()

    aim_strs = [[0] * n_qubits for _ in range(n_qubits)]
    
    for i in range(len(aim_strs)):
        aim_strs[i][i] = 1
    aim_strs += [[0] * n_qubits]


    if equal_shot_distribution:
        confirmation_shots = n_shots / (2 * k) # Half of all shots on confirmation
        model_shots = n_shots / (2 * len(aim_strs)) # Half of all shots on model building
    
    aim_results = {}
    for inversion_arr in aim_strs:
    
        # Build and execute AIM circuits
        tmp_circuit = copy.deepcopy(circuit)
        tmp_circuit = design_circuit(n_qubits, inversion_arr, circuit=tmp_circuit)
        job = qiskit.execute(tmp_circuit, backend, shots=model_shots)
        results = job.result().get_counts()

        # Simulated Noisy Measurement
        if probs is not None:
            noisy_measurement = measurement_error(results, n_qubits=n_qubits, probs=probs)
            results = sample_distribution(noisy_measurement, model_shots)

        # Invert AIM Strings
        aim_result = {}
        for count in results:
            count_arr = ''.join(map(str, [i ^ j for i, j in zip(inversion_arr, map(int, list(count[::-1])))]))

            if count_arr in aim_result:
                aim_result[count_arr] += results[count]
            else:
                aim_result[count_arr] = results[count]

        aim_results[''.join(map(str, inversion_arr))] = aim_result
    
    # Join across inversion strings
    likelihoods = {}
    for res in aim_results:
        for state in aim_results[res]:
            if state in likelihoods:
                likelihoods[state] += aim_results[res][state]
            else:
                likelihoods[state] = aim_results[res][state]
    
    # Select top k strings
    top_k = []
    for i in range(k):
        top = max(likelihoods.items(), key=lambda i: i[1])
        top_k.append(top[0])
        likelihoods.pop(top[0])
    

    # Run confirmation shots for statistics
    tmp_circuits = [copy.deepcopy(circuit) for _ in range(k)]
    tmp_circuits = [design_circuit(n_qubits, i, circuit=t) for i, t in zip(top_k, tmp_circuits)]

    # Run final
    aim_result = {}
    for circ, inversion_arr in zip(tmp_circuits, top_k):

        job = qiskit.execute(circ, backend, shots=confirmation_shots)
        results = job.result().get_counts()

        # Simulated Noisy Measurement
        if probs is not None:
            noisy_measurement = measurement_error(results, n_qubits=n_qubits, probs=probs)
            results = sample_distribution(noisy_measurement, confirmation_shots)

        for count in results:
            # Invert measurement results using k strings
            count_arr = ''.join(
                map(str, [i ^ j for i, j in zip(map(int, list(inversion_arr)), map(int, list(count[::-1])))])
            )

            if count_arr in aim_result:
                aim_result[count_arr] += results[count]
            else:
                aim_result[count_arr] = results[count]

    return aim_result