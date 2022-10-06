from datetime import datetime
import qiskit
from qiskit.providers.fake_provider import fake_backend
from qiskit.providers.models import BackendProperties, QasmBackendConfiguration

from functools import partial
from numpy import random


basis_gates_1q = ["id", "rz", "sx", "x"]
basis_gates_2q = ["cx"]

supported_instructions = ["measure", "u3", "setf", "x", "delay", "id", "acquire", "sx", "u2", "u1", "cx", "shiftf", "play", "rz", "reset"]
basis_gates = basis_gates_1q + basis_gates_2q

# Wrapper for const properties
def const(x):
    def fn(*args, **kwargs):
        return x
    return fn

def uniform_random(low, high, *args, size=None, **kwargs):
    return low + (high - low) * random.random(size=size)
    

def qubit_property_builder(name, value, unit, *args, curr_time=None, **kwargs):
    return {"date": curr_time, "unit": unit, "value": value(*args, **kwargs), "name": name}

def gate_property_builder(name, error, duration, qubits, *args, curr_time=None, **kwargs):    
    return {"parameters": [
                qubit_property_builder("gate_error", error, '', qubits, *args, curr_time=curr_time, **kwargs),
                qubit_property_builder("gate_length", duration, 'ns', qubits, *args, curr_time=curr_time, **kwargs)
                ],
            "qubits": qubits,
            "gate" : name,
            "name" : '_'.join([name, *map(str, qubits)])
           }

def generate_configuration(name, n_qubits, coupling_map):
    return QasmBackendConfiguration(
            backend_name=name,
            n_qubits=n_qubits,
            basis_gates=basis_gates,
            simulator=False,
            local=True,
            coupling_map=coupling_map,
            backend_version="0.0.1",
            gates=[],
            open_pulse=False,
            conditional=False,
            memory=False,
            max_shots=float('inf'),
            max_experiments=float('inf'),
            supported_instructions=supported_instructions,
            multi_meas_enabled=False
        )

def generate_properties(name, 
                        n_qubits, 
                        coupling_map, 
                        errors_1q = const(0.001), # TODO GET LIMA DATA
                        errors_2q = const(0.01),  # TODO GET LIMA DATA
                        meas_errors01 = partial(uniform_random, 0.02, 0.08),  # TODO GET LIMA DATA
                        meas_errors10 = partial(uniform_random, 0.02, 0.08),  # TODO GET LIMA DATA
                       t1 = const(100),
                       t2 = const(50),
                       freq = const(5),
                       readout = partial(uniform_random, 0.02, 0.08),
                       gate_duration = const(8),
                       curr_time = str(datetime.now())):
    properties = {"backend_version": "0.0.1", "general": [], "last_update_date": curr_time, "backend_name": name}
    
    qubit_properties_template = [
        ['T1', t1, 'µs'],
        ['T2', t2, 'µs'],
        ['frequency', freq, 'GHz'],
        ['readout', readout, ''],
        ["prob_meas0_prep1", meas_errors01, ''],
        ["prob_meas1_prep0", meas_errors10, ''],
    ]
        
    gate_properties_1q_template = [[gate, errors_1q, const(8)] for gate in basis_gates_1q]
    gate_properties_2q_template = [[gate, errors_2q, const(128)] for gate in basis_gates_2q]
    
    qubit_properties = []
    for qubit in range(n_qubits):
        qubit_property = []
        for template in qubit_properties_template: 
            qubit_property.append(qubit_property_builder(*template, qubit, curr_time=curr_time))
        qubit_properties.append(qubit_property)
    
    gate_properties = []
    for qubit in range(n_qubits):
        for gate in gate_properties_1q_template:
            gate_properties.append(
                gate_property_builder(*gate, [qubit], curr_time=curr_time)
            )    
    
    for pair in coupling_map:
        for gate in gate_properties_2q_template:
            gate_properties.append(
                gate_property_builder(*gate, pair, curr_time=curr_time)
            )
            
    properties["qubits"] = qubit_properties
    properties["gates"] = gate_properties
    return properties

class FakeBackendWrapper(fake_backend.FakeBackend):
    """A fake 16 qubit fully connected backend."""

    def __init__(self, name, n_qubits, coupling_map):
        self._name = name
        self.n_qubits = n_qubits
        self._coupling_map = coupling_map
        
        self._configuration = generate_configuration(self._name, self.n_qubits, self._coupling_map)
        
        properties = generate_properties(self.name, self.n_qubits, self._coupling_map)
        self._properties = BackendProperties.from_dict(properties)
        
        super().__init__(self._configuration)
        self._target = None
        self.sim = None
        
    def properties(self):
        return self._properties
    
    @staticmethod
    def cmap_append(lst, a, b):
        lst.append([a, b])
        lst.append([b, a])

class FullyConnected(FakeBackendWrapper):
    """A fake fully connected backend."""

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self._coupling_map = self.gen_coupling_map(self.n_qubits)
        super().__init__("full", n_qubits, self._coupling_map)
        
    @staticmethod
    def gen_coupling_map(n_qubits):
        return [[i, j] for i in range(n_qubits) for j in range(n_qubits) if i != j] 

class Hexagonal16(FakeBackendWrapper):
    """A fake 16 qubit fully connected backend."""

    def __init__(self):
        self.n_qubits = 16
        self._coupling_map = self.gen_coupling_map()
        super().__init__("hex", self.n_qubits, self._coupling_map)

    
    def gen_coupling_map(self):
        return [
        [0, 1], [1, 0],
        [0, 2], [2, 0],
        [1, 3], [3, 1],
        [2, 4], [4, 2],
        [4, 5], [5, 4],
        [3, 5], [5, 3],
        [3, 13], [13, 3],
        [13, 14], [14, 13],
        [14, 15], [15, 14],
        [10, 15], [15, 10],
        [10, 5], [5, 10],
        [10, 12], [12, 10],
        [11, 12], [12, 11],
        [9, 11], [11, 9],
        [9, 4], [4, 9],
        [9, 8], [8, 9],
        [8, 7], [7, 8],
        [7, 6], [6, 7],
        [6, 2], [2, 6],
    ]

class Hexagonal(FakeBackendWrapper):
    """A fake Hexagonal backend."""

    def __init__(self, n_rows, n_columns):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_qubits = None
        self._coupling_map = None
        self.gen_coupling_map()
        super().__init__("hex", self.n_qubits, self._coupling_map)

    
    def gen_coupling_map(self):
        q_rows = self.n_rows * 3
        q_cols = self.n_columns * 2
        self.n_qubits = q_rows * q_cols
        self._coupling_map = []
        
        # Vertical joins        
        for i in range(q_rows):
            for j in range(q_cols):
                
                # Vertical edges
                if (i < q_rows - 1):
                    super().cmap_append(
                        self._coupling_map,
                        self.pti(i, j),
                        self.pti(i + 1, j)
                    )
                    
                # Horizontal edges
                if ((i + j) % 2) == 0 and j < q_cols - 1:
                    super().cmap_append(
                        self._coupling_map,
                        self.pti(i, j),
                        self.pti(i, j + 1)
                    )
    
    def pti(self, i, j):
        '''
            Turns a set of coordinates into an integer
        '''
        return i * self.n_columns * 2 + j
                
class Grid(FakeBackendWrapper):
    """A Gridded backend."""

    def __init__(self, n_rows, n_columns):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_qubits = None
        self._coupling_map = None
        self.gen_coupling_map()
        super().__init__("square", self.n_qubits, self._coupling_map)

    
    def gen_coupling_map(self):
        self.n_qubits = self.n_rows * self.n_columns
        self._coupling_map = []
        
        # Vertical joins        
        for i in range(self.n_rows):
            for j in range(self.n_columns):
                if (i < self.n_rows - 1):
                    super().cmap_append(
                        self._coupling_map,
                        self.pti(i, j),
                        self.pti(i + 1, j)
                    )
                if j < self.n_columns - 1:
                    super().cmap_append(
                        self._coupling_map,
                        self.pti(i, j),
                        self.pti(i, j + 1)
                    )
    
    def pti(self, i, j):
        '''
            Turns a set of coordinates into an integer
        '''
        return i + j * self.n_rows
