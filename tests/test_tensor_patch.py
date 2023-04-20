import unittest as pyunit

from PatchedMeasCal.edge_bfs import CouplingMapGraph
from PatchedMeasCal.tensor_patch_cal import TensorPatchFitter

import qiskit
from qiskit.providers.fake_provider import FakeVigo, FakeTokyo

class TensorPatchTest(pyunit.TestCase):

    def test_build(self):
        '''
            Loads a real backend object and checks that duplicate edge directions are culled
        '''
        backend = FakeVigo()
        tpf = TensorPatchFitter(backend)
        tpf.build()

    def test_apply(self):
        backend = FakeVigo()
        tpf = TensorPatchFitter(backend)
        tpf.build()

        results_vec = {'11111':10, '00001':12}
        tpf.apply(results_vec)

   def test_partial(self):
        backend = FakeVigo()
        tpf = TensorPatchFitter(backend)
        tpf.build()

        results_vec = {'1111':10, '0001':12}
        tpf.apply(results_vec, participating_qubits=[0, 1, 2, 3])

     def test_large(self):
         backend = FakeTokyo()
         tpf = TensorPatchFitter(backend)
         tpf.build()

         results_vec = {'11111':10, '00001':12}
         tpf.apply(results_vec)

