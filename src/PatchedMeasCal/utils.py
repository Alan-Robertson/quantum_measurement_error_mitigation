from math import floor
import time

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
            if self.invoked == 0:
                self.start_time = time.time()
                return 'ETA: '
            else:
                curr_time = time.time()
                time_per_unit = (curr_time - self.start_time) / self.invoked
                est_time_left = int((self.n_calls - self.invoked) * time_per_unit)
                return 'ETA: {}s'.format(est_time_left)
        return ''
                







