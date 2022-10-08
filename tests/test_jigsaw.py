import unittest as pyunit

from PatchedMeasCal import jigsaw

class StatePrepTest(pyunit.TestCase):

    def test_convolve(self):
        '''
        Testing Bayes update using tables from the paper
        '''
        global_pmf_table = {
        '000':0.1,
        '001':0.10,
        '010':0.15,
        '011':0.15,
        '100':0.10,
        '101':0.05,
        '110':0.15,
        '111':0.2}

        local_table = {'00':0.1, '01':0.1, '10':0.2, '11':0.6}

        local_pair = [1, 2] # Indices

        updated_table = jigsaw.convolve(global_pmf_table, local_table, local_pair)
        
        expected_table = {
        '000':0.05,
        '001':0.07,
        '010':0.13,
        '011':0.64,
        '100':0.05,
        '101':0.04,
        '110':0.13,
        '111':0.86
        }

        norm_val = sum(expected_table.values())
        for i in expected_table:
            expected_table[i] /= norm_val

        for i in expected_table:
            assert(abs(expected_table[i] - updated_table[i]) < 0.05)

            



        
    
