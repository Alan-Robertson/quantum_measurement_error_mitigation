import unittest as pyunit
from PatchedMeasCal.edge_bfs import CouplingMapGraph

class EdgeBFSTest(pyunit.TestCase):


    def all_edges(self, cmap, patches):
        '''Checks that all measured edges are in the patches'''
        flat_patches = []
        for patch in patches:
            flat_patches += patch

        for edge in cmap:
            assert(edge in flat_patches)
        return True


    def test_loop(self):
        cmap = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
        cmg = CouplingMapGraph(cmap)
        patches = cmg.edge_patches()

        assert(self.all_edges(cmap, patches))


        
    
