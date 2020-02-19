from typing import Callable, Dict
from ..data.access import DataAccess, mens_access, womens_access


class Features():
    def __init__(self):
        self.features = {}  # type: Dict[str,Callable]

    def register(self, f: Callable):
        self.features[f.__name__] = f

    def run(self, access: DataAccess):
        return {name: f(access) for name, f in self.features.items()}


registry = Features()

from . import simple
