import pandas as pd

from . import pf
from ..data.access import DataAccess
from ..data.processed import player_scoring_df


@pf.register
def player_scoring(access: DataAccess) -> pd.DataFrame:
    p_scoring_df = player_scoring_df(access=access)
    return p_scoring_df
