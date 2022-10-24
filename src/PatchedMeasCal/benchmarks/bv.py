import qiskit

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