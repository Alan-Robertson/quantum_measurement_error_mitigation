from math import floor

'''
    A few utility functions
'''

def vprint(verbose, string):
    if verbose:
        print(string)

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

class Progressbar():
    
    def __init__(
            self, 
            n_ticks, # Number of ticks to print
            n_calls, # Number of times you expect to call this progress bar
            name='', # Name of bar to display
            ticker='=', # Ticker symbol(s)
            ticker_head='>', # Head of the ticker
            ticker_blank=' '): # Unticked symbol(s)
        self.n_ticks = n_ticks
        self.n_calls = n_calls
        self.name = name
        self.ticker=ticker
        self.ticker_head = ticker_head
        self.ticker_blank = ticker_blank
        self.invoked = -1 # Number of times it has been invoked

    def tick(self, message=''):
        ticker = floor(self.invoked / self.n_calls * self.n_ticks) * self.ticker
        blank = (self.n_ticks - len(ticker) - 1) * self.ticker_blank
        head = [self.ticker_head, ''][len(ticker) >= self.n_ticks]
        self.invoked += 1 

        if self.invoked > self.n_calls:
            return

        if self.invoked + 1 >= self.n_calls:
            self.invoked += 1

        fstring = "\r{name} : [{ticker}{head}{blank}] {pct:.1f}% {msg}".format(
                name = self.name,
                ticker = ticker,
                head = head,
                blank = blank, 
                pct = 100 * min(1, self.invoked / self.n_calls),
                msg = message
                )
        print(fstring, end='', flush=True)
        
        if self.invoked >= self.n_calls:
            print()

    def __call__(self, *args, **kwargs):
        return self.tick(*args, **kwargs)