from . import mens_features, womens_features
from ..data.access import DataAccess
from ..data.processed import regular_season_compact_team_results_df
import pandas as pd


@mens_features.register
def mens_score_difference(access: DataAccess) -> pd.DataFrame:
    ctr_df = regular_season_compact_team_results_df(access=access)
    return (ctr_df.Score - ctr_df.OtherScore).rename('ScoreDifference')
