from typing import Callable, Dict, Union
from ..data.access import DataAccess
import pandas as pd
from tqdm import tqdm


class Features():
    def __init__(self):
        self.features = {}  # type: Dict[str,Callable[[DataAccess], Union[pd.DataFrame, pd.Series]]]

    def register(self, f: Callable[[DataAccess], Union[pd.DataFrame, pd.Series]]):
        self.features[f.__name__] = f
        return f

    def run(self, *feature_names, access: DataAccess) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        to_run = ((feature_name, self.features.get(feature_name)) for feature_name in
                  feature_names) if feature_names else self.features.items()
        return {name: f(access) for name, f in
                tqdm(iterable=to_run, desc=f'Computing "{access.prefix}" prefixed Features', leave=True)}


tf = Features()
pf = Features()
tpf = Features()

from . import team_features
from . import player_features
from . import team_player_features
