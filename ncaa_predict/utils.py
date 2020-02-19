from typing import Callable


def memoize(f: Callable):
    memo = {}

    def _f(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return _f
