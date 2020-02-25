from . import registry
from ..data.access import DataAccess
from ..data.processed import regular_season_compact_team_results_df, team_format_indices
import pandas as pd
from ..utils import memoize


@registry.register
def score_difference(access: DataAccess) -> pd.Series:
    ctr_df = regular_season_compact_team_results_df(access)
    return (ctr_df.Score - ctr_df.OtherScore).rename('ScoreDifference')


@registry.register
def win(access: DataAccess) -> pd.Series:
    ctr_df = regular_season_compact_team_results_df(access)
    return (((ctr_df.Score > ctr_df.OtherScore).astype(int) * 2) - 1).rename('Win')


@registry.register
def streak(access: DataAccess) -> pd.Series:
    _win_df = win(access).reset_index()
    _shifted_win_df = _win_df.shift(1)

    _win_df['StreamNum'] = ((_win_df.Win != _shifted_win_df.Win)
                            | (_win_df.TeamID != _shifted_win_df.TeamID)
                            | (_win_df.Season != _shifted_win_df.Season)).cumsum()
    _win_df['Streak'] = _win_df.groupby('StreamNum').Win.cumsum()
    _win_df.set_index(team_format_indices, inplace=True)
    _win_df.sort_index(inplace=True)
    return _win_df.Streak


@registry.register
def home_advantage(access: DataAccess) -> pd.Series:
    ctr_df = regular_season_compact_team_results_df(access).copy()
    ctr_df.loc[ctr_df.Loc == 'H', 'HomeAdvantage'] = 1
    ctr_df.loc[ctr_df.Loc == 'N', 'HomeAdvantage'] = 0
    ctr_df.loc[ctr_df.Loc == 'A', 'HomeAdvantage'] = -1
    return ctr_df.HomeAdvantage


@registry.register
@memoize
def rest_days(access: DataAccess) -> pd.Series:
    ctr_df = regular_season_compact_team_results_df(access).reset_index()
    first_season = min(ctr_df.Season)
    ctr_df['OverallDayNum'] = ((ctr_df.Season - first_season) * 365) + ctr_df.DayNum
    ctr_df.set_index(team_format_indices, inplace=True)
    ctr_df.sort_index(inplace=True)
    ctr_df['RestDays'] = ctr_df.groupby('TeamID').OverallDayNum.diff()
    return ctr_df.RestDays


def _rest_days_with_maximum(access: DataAccess, maximum: int) -> pd.Series:
    rest = rest_days(access).rename(f'RestDaysMax{maximum}')
    rest.where(cond=rest < maximum, other=maximum, inplace=True)
    return rest


@registry.register
def rest_days_14_max(access: DataAccess):
    return _rest_days_with_maximum(access=access, maximum=14)


@registry.register
def rest_days_7_max(access: DataAccess):
    return _rest_days_with_maximum(access=access, maximum=7)


@registry.register
def rest_days_3_max(access: DataAccess):
    return _rest_days_with_maximum(access=access, maximum=3)
