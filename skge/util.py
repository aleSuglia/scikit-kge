import numpy as np
from numpy.fft import fft, ifft
import scipy.sparse as sp
import functools
import collections
import sys
import time


def cconv(a, b):
    """
    Circular convolution of vectors

    Computes the circular convolution of two vectors a and b via their
    fast fourier transforms

    a \ast b = \mathcal{F}^{-1}(\mathcal{F}(a) \odot \mathcal{F}(b))

    Parameter
    ---------
    a: real valued array (shape N)
    b: real valued array (shape N)

    Returns
    -------
    c: real valued array (shape N), representing the circular
       convolution of a and b
    """
    return ifft(fft(a) * fft(b)).real


def ccorr(a, b):
    """
    Circular correlation of vectors

    Computes the circular correlation of two vectors a and b via their
    fast fourier transforms

    a \ast b = \mathcal{F}^{-1}(\overline{\mathcal{F}(a)} \odot \mathcal{F}(b))

    Parameter
    ---------
    a: real valued array (shape N)
    b: real valued array (shape N)

    Returns
    -------
    c: real valued array (shape N), representing the circular
       correlation of a and b
    """

    return ifft(np.conj(fft(a)) * fft(b)).real


def grad_sum_matrix(idx):
    uidx, iinv = np.unique(idx, return_inverse=True)
    sz = len(iinv)
    M = sp.coo_matrix((np.ones(sz), (iinv, np.arange(sz)))).tocsr()
    # normalize summation matrix so that each row sums to one
    n = np.array(M.sum(axis=1))
    #M = M.T.dot(np.diag(n))
    return uidx, M, n


def unzip_triples(xys, with_ys=False):
    xs, ys = list(zip(*xys))
    ss, os, ps = list(zip(*xs))
    if with_ys:
        return np.array(ss), np.array(ps), np.array(os), np.array(ys)
    else:
        return np.array(ss), np.array(ps), np.array(os)


def to_tensor(xs, ys, sz):
    T = [sp.lil_matrix((sz[0], sz[1])) for _ in range(sz[2])]
    for i in range(len(xs)):
        i, j, k = xs[i]
        T[k][i, j] = ys[i]
    return T


def init_nvecs(xs, ys, sz, rank, with_T=False):
    from scipy.sparse.linalg import eigsh

    T = to_tensor(xs, ys, sz)
    T = [Tk.tocsr() for Tk in T]
    S = sum([T[k] + T[k].T for k in range(len(T))])
    _, E = eigsh(sp.csr_matrix(S), rank)
    if not with_T:
        return E
    else:
        return E, T


class memoized(object):
    '''
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    see https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    '''

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncachable, return direct function application
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            val = self.func(*args)
            self.cache[args] = val
            return val

    def __repr__(self):
        '''return function's docstring'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''support instance methods'''
        return functools.partial(self.__call__, obj)

    
class Progbar(object):
    def __init__(self, target, width=30, verbose=1, interval=0.01):
        """
            @param target: total number of steps expected
            @param interval: minimum visual progress update interval (in seconds)
        """
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], force=False):
        """
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
            @param force: force visual progress update
        """
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return

            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if type(self.sum_values[k]) is list:
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)

