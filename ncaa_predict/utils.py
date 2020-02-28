def memoize(f):
    memo = {}

    def helper(*args, **kwargs):
        x = frozenset((*args, *(kwargs.items())))
        if x not in memo:
            memo[x] = f(*args, **kwargs)
        return memo[x]

    return helper
