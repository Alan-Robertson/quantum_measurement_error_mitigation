'''
    A few utility functions
'''

def normalise(x):
    '''
        Normalise the partial trace of a calibration matrix
    '''
    for i in range(x.shape[1]):
        tot = sum(x[:, i])
        if tot != 0:
            x[:, i] /= tot
    return x

def f_dims(n):
    '''
        Dimension ordering for n qubits
    '''
    return [[2 for i in range(n)]] * 2