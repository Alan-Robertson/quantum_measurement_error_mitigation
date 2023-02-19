import qiskit
import copy

from PatchedMeasCal.edge_bfs import cmap_djikstra, cmap_shortest_path

def bv_circuit(bv_string, n_qubits):
    bv_circuit = qiskit.QuantumCircuit(n_qubits, n_qubits)
    
    for i in range(n_qubits):
        bv_circuit.h(i)
        
    bv_circuit.z(n_qubits - 1)
    
    bv_circuit.barrier()
    
    for i in range(n_qubits -1):
        if int(bv_string[i]) == 1:
            bv_circuit.cx(i, n_qubits - 1)
    
    
    bv_circuit.barrier()
    
    for i in range(n_qubits):
        bv_circuit.h(i)

    bv_circuit.measure(
        list(range(n_qubits)), 
        list(range(n_qubits))
    )
    return bv_circuit

def bv_circuit_cmap(bv_string, n_qubits, backend):
    coupling_map = backend.configuration().coupling_map

    circ = qiskit.QuantumCircuit(n_qubits, n_qubits)
    
    for i in range(n_qubits):
        circ.h(i)
    circ.z(n_qubits - 1)
    circ.barrier()
    
    target = [int(i) for i in bv_string] + [1]
    current = [0] * (n_qubits - 1) + [1]
    
    cnot_chain = cnot_network(current, target, coupling_map, n_qubits)
    
    for i in cnot_chain:
        circ.cnot(*i[::-1])   
    
    circ.barrier()
    
    for i in range(n_qubits):
        circ.h(i)
   
    circ.measure(
        list(range(n_qubits)), 
        list(range(n_qubits))
    )

    return circ


def cnot_network(initial_state, target_state, coupling_map, n_qubits):
    network = []

    initial_state = copy.deepcopy(initial_state)

    distance_map = cmap_djikstra(coupling_map, n_qubits)

    while initial_state != target_state:
        mask = [i ^ j for i, j in zip(initial_state, target_state)]

        # Get longest path distance remaining in the stack
        shortest_path = [float('inf'), None]
        for i in range(n_qubits):
            if mask[i] == 1:
                for j in range(n_qubits):
                    if initial_state[j] == 1 and i != j:
                        #shortest_path = cmap_shortest_path(j, i, distance_map, coupling_map)
                        if shortest_path[0] > distance_map[i][j]:
                            shortest_path = [distance_map[i][j], [j, i]]


        path = cmap_shortest_path(*shortest_path[1], distance_map, coupling_map)
        for i, j in zip(path[:-1], path[1:]):
            network.append([i, j])
            initial_state[j] ^= initial_state[i]

    return network





