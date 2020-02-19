from . import registry
from ..data.access import DataAccess
from ..data.processed import regular_season_compact_team_results_df
from ..utils import memoize
import pandas as pd


@registry.register
def score_difference(access: DataAccess) -> pd.Series:
    ctr_df = regular_season_compact_team_results_df(access)
    return (ctr_df.Score - ctr_df.OtherScore).rename('ScoreDifference')


@registry.register
def win(access: DataAccess) -> pd.Series:
    ctr_df = regular_season_compact_team_results_df(access)
    return (ctr_df.Score > ctr_df.OtherScore).rename('Win')
