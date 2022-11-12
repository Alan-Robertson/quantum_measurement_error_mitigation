from math import floor
import time
import copy

'''
    A few utility functions
'''

def vprint(verbose, string):
    if verbose:
        print(string)



def vProgressbar(verbose, *args, **kwargs):
    if verbose:
        return Progressbar(*args, **kwargs)
    return None

def vtick(verbose, pb, *args, **kwargs):
    if verbose:
        pb.tick(*args, **kwargs)

def vflush(verbose):
    if verbose:
        print()

def norm_results_dict(results):
    norm_val = sum(results.values())
    for i in results: 
        results[i] /= norm_val

def dict_distance(noise_free_results, error_prone_results):
    noise_free_results = copy.deepcopy(noise_free_results)
    error_results = copy.deepcopy(error_prone_results)
    norm_results_dict(noise_free_results)
    norm_results_dict(error_results)

    dist = 0
    for res in noise_free_results:
        if res in error_prone_results:
            dist += abs(noise_free_results[res] - error_results[res])
        else:
            dist += noise_free_results[res]
    return dist

class Progressbar():
    
    def __init__(
            self, 
            n_ticks, # Number of ticks to print
            n_calls, # Number of times you expect to call this progress bar
            name='', # Name of bar to display
            ticker='=', # Ticker symbol(s)
            ticker_head='>', # Head of the ticker
            ticker_blank=' ',
            eta=True): # Unticked symbol(s)
        self.n_ticks = n_ticks
        self.n_calls = n_calls
        self.name = name
        self.ticker=ticker
        self.ticker_head = ticker_head
        self.ticker_blank = ticker_blank
        self.invoked = -1 # Number of times it has been invoked
        self.p_count = 0
        self.start_time = None
        self.eta = eta

    def tick(self, message=''):

        if self.invoked > 1:
            # Clear anything left from prev round
            print('\b' * self.p_count, end='', flush=True)
        
        if self.invoked > self.n_calls: # Invoked too many times
            return
        self.invoked += 1        

        if self.invoked + 1 >= self.n_calls: # Final Call
            self.invoked = self.n_calls

        ticker = floor(self.invoked / self.n_calls * self.n_ticks) * self.ticker
        blank = (self.n_ticks - len(ticker) - 1) * self.ticker_blank
        head = [self.ticker_head, ''][len(ticker) >= self.n_ticks]
        eta = self.eta_calc()

        fstring = "\r{name} : [{ticker}{head}{blank}] {pct:.1f}% {msg} {eta}".format(
                name = self.name,
                ticker = ticker,
                head = head,
                blank = blank, 
                pct = 100 * min(1, self.invoked / self.n_calls),
                msg = message,
                eta = eta
                )
        print(fstring, end='', flush=True)
        self.p_count = len(fstring) - 1
        
        if self.invoked >= self.n_calls: # Final Call, print newline
            print()

    def __call__(self, *args, **kwargs):
        return self.tick(*args, **kwargs)

    def eta_calc(self):
        if self.eta:
            if self.invoked == 0 or self.start_time is None:
                self.start_time = time.time()
                return 'ETA: '
            else:
                curr_time = time.time()
                time_per_unit = (curr_time - self.start_time) / self.invoked
                est_time_left = int((self.n_calls - self.invoked) * time_per_unit)
                return 'ETA: {}s'.format(est_time_left)
        return ''
                

def f_dims(n:int):
    '''
        Dimension ordering for n qubits
    '''
    return [[2 for i in range(n)]] * 2

def normalise(x): # Array
    '''
        Normalise the partial trace of a calibration matrix
    '''
    for i in range(x.shape[1]):
        tot = sum(x[:, i])
        if tot != 0:
            x[:, i] /= tot
    return x

def list_fold(lst, fold_length):
    for i in zip(*[lst[i::fold_length] for i in range(fold_length)]):
        yield i
    return



