import unittest as pyunit

from PatchedMeasCal.edge_bfs import CouplingMapGraph
from PatchedMeasCal.tensor_patch_cal import TensorPatchFitter

from qiskit.providers.fake_provider import FakeVigo

class BackendTest(pyunit.TestCase):


    def test_build(self):
        '''
            Loads a real backend object and checks that duplicate edge directions are culled
        '''
        backend = FakeVigo()
        tpf = TensorPatchFitter(backend)
        tpf.build()
        #tpf._coupling_graph = CouplingMapGraph(backend.configuration().coupling_map)
        #tpf.construct_edge_calibrations(4000)

    def test_meas_fitter(self):
        '''
            Loads a real backend object and checks that duplicate edge directions are culled
        '''
        backend = FakeVigo()
        tpf = TensorPatchFitter(backend)
        tpf.build()
        tpf.build_meas_fitter()

        



        
    
