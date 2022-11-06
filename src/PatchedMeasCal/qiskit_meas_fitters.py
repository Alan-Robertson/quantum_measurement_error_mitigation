import qiskit
from qiskit.ignis.mitigation.measurement import complete_meas_cal, tensored_meas_cal, CompleteMeasFitter, TensoredMeasFitter

def qiskit_full(backend, n_qubits, n_shots, probs=None):
    # Half shots on building the model, half on calibration
    # For a fair test uncomment the next line, however for large n this will rapidly become useless
    n_shots_qiskit_full = n_shots #// (2 ** (n_qubits - 1)) 

    qr = qiskit.QuantumRegister(n_qubits)
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    t_qc = qiskit.transpile(meas_calibs, backend)
    cal_results = qiskit.execute(t_qc, backend, shots=n_shots_qiskit_full).result()
    cal_res_measurement_error(cal_results, probs)
        
    meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
    full_filter = meas_fitter.filter
    return full_filter

def qiskit_linear(backend, n_qubits, n_shots, probs=None):
    # Build Calibration Matrix
    mit_pattern = [[i] for i in range(n_qubits)]

    # Half shots on building the model, half on calibration
    n_shots_qiskit_partial = n_shots // (n_qubits / 2)

    qr = qiskit.QuantumRegister(n_qubits)
    meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')
    t_qc = qiskit.transpile(meas_calibs, backend)
    cal_results = qiskit.execute(t_qc, backend, shots=n_shots).result()
    cal_res_measurement_error(cal_results, probs)
    
    meas_fitter = TensoredMeasFitter(cal_results, state_labels, circlabel='mcal')
    linear_fitter = meas_fitter.filter
    return linear_fitter