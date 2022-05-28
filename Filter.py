from numpy.core import empty, arange


def filter_func(n, d=1.0):
    val = 1.0 / (n * d)
    results = empty(n, int)
    N = (n - 1) // 2 + 1
    p1 = arange(0, N, dtype=int)
    results[:N] = p1
    p2 = arange(-(n // 2), 0, dtype=int)
    results[N:] = p2
    return results * val
