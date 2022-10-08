import unittest as pyunit

from PatchedMeasCal.edge_bfs import CouplingMapGraph
from PatchedMeasCal.tensor_patch_cal import TensorPatchFitter

from PatchedMeasCal import state_prep_circuits

class StatePrepTest(pyunit.TestCase):

    def test_res_to_vec(self):
        '''
            Loads a real backend object and checks that duplicate edge directions are culled
        '''
        r = {'11':10, '00':9, '10':8, '01':7}
        assert(list(state_prep_circuits.res_to_vec(r)) == [9, 8, 7, 10])

    def test_plus_state(self):
        '''
            Loads a real backend object and checks that duplicate edge directions are culled
        '''
        r = {'111':10, '000':10}
        assert(state_prep_circuits.plus_state_dist(r) == 0.0)


        



        
    
