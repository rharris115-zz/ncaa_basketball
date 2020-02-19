from typing import Callable, Dict
from ..data.access import DataAccess, mens_access, womens_access


class Features():
    def __init__(self, access: DataAccess):
        self.features = {}  # type: Dict[str,Callable]
        self.access = access

    def register(self, f: Callable):
        self.features[f.__name__] = f

    def run(self):
        return {name: f(self.access) for name, f in self.features.items()}


mens_features = Features(access=mens_access)
womens_features = Features(access=womens_access)

from . import simple
