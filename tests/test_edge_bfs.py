import unittest as pyunit
from PatchedMeasCal.edge_bfs import CouplingMapGraph

from qiskit.providers.fake_provider import FakeVigo

class EdgeBFSTest(pyunit.TestCase):
    '''
        Tests for the Edge BFS
    '''

    def all_edges(self, cmap, patches):
        '''
            A set of common tests for the edge bfs
            - Checks that all edges in the coupling map are in the patches
            - Checks that no edge is included with its obverse
        '''
        flat_patches = []
        for patch in patches:
            flat_patches += patch

        # Either the edge or its inverse should be in the flattened patches
        for edge in cmap:
            assert(edge in flat_patches or edge[::-1] in flat_patches) 
            
        # No edges should be duplicated and reversed
        for edge in flat_patches:
            assert(edge[::-1] not in flat_patches)

        return True

    def test_duplicates(self):
        '''
            Loads a real backend object and checks that duplicate edge directions are culled
        '''
        backend = FakeVigo()
        cmap = backend.configuration().coupling_map
        cmg = CouplingMapGraph(cmap)
        patches = cmg.edge_patches()
        assert(self.all_edges(cmap, patches))


    def test_loop(self):
        '''
            Simple test with a loop architecture
        '''
        cmap = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
        cmg = CouplingMapGraph(cmap)
        patches = cmg.edge_patches()
        assert(self.all_edges(cmap, patches))


        
    
