from . import registry
from ..data.access import DataAccess
from ..data.processed import regular_season_compact_team_results_df
import pandas as pd
import numpy as np


@registry.register
def score_difference(access: DataAccess) -> pd.Series:
    ctr_df = regular_season_compact_team_results_df(access)
    return (ctr_df.Score - ctr_df.OtherScore).rename('ScoreDifference')


@registry.register
def win(access: DataAccess) -> pd.Series:
    ctr_df = regular_season_compact_team_results_df(access)
    return (ctr_df.Score > ctr_df.OtherScore).rename('Win')


def _rest_days(access: DataAccess, maximum: int) -> pd.Series:
    ctr_df = regular_season_compact_team_results_df(access).reset_index()
    ctr_df['Day'] = (ctr_df.Season * 365) + ctr_df.DayNum
    ctr_df['RestDays'] = ctr_df.groupby('TeamID').Day.diff().fillna(maximum)
    ctr_df.RestDays = np.where(ctr_df.RestDays > maximum, maximum, ctr_df.RestDays)
    return ctr_df['RestDays']


@registry.register
def rest_days_two_weeks_max(access: DataAccess):
    return _rest_days(access=access, maximum=14)


@registry.register
def rest_days_one_week_max(access: DataAccess):
    return _rest_days(access=access, maximum=7)


@registry.register
def rest_days_three_days_max(access: DataAccess):
    return _rest_days(access=access, maximum=3)
