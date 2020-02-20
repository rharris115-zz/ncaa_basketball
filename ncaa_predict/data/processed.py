from .access import DataAccess
from ..utils import memoize
from typing import Dict
import pandas as pd


@memoize
def regular_season_compact_team_results_df(access: DataAccess) -> pd.DataFrame:
    compact_results_df = regular_season_compact_results_df(access)
    return _compact_team_results_df(compact_results_df=compact_results_df)


@memoize
def tourney_season_compact_team_results_df(access: DataAccess) -> pd.DataFrame:
    compact_results_df = tourney_season_compact_results_df(access)
    return _compact_team_results_df(compact_results_df=compact_results_df)


@memoize
def all_team_results_df(access: DataAccess) -> pd.DataFrame:
    _regular_season_compact_team_results_df = regular_season_compact_team_results_df(access).copy()
    _tourney_season_compact_team_results_df = tourney_season_compact_team_results_df(access).copy()

    _regular_season_compact_team_results_df['Tourney'] = False
    _tourney_season_compact_team_results_df['Tourney'] = True

    _compact_team_results_df = _regular_season_compact_team_results_df \
        .append(_tourney_season_compact_team_results_df)
    _compact_team_results_df.sort_index(inplace=True)
    return _compact_team_results_df


@memoize
def regular_season_compact_results_df(access: DataAccess) -> pd.DataFrame:
    compact_results_df = access.regular_season_compact_results_df()
    team_names_by_id = access.teams_df().set_index('TeamID').TeamName.to_dict()
    return _ccompact_results_with_team_names_df(team_names_by_id=team_names_by_id,
                                                compact_results_df=compact_results_df)


@memoize
def tourney_season_compact_results_df(access: DataAccess) -> pd.DataFrame:
    compact_results_df = access.tourney_compact_results_df()
    team_names_by_id = access.teams_df().set_index('TeamID').TeamName.to_dict()
    return _ccompact_results_with_team_names_df(team_names_by_id=team_names_by_id,
                                                compact_results_df=compact_results_df)


def all_season_compact_results_df(access: DataAccess) -> pd.DataFrame:
    _regular_season_compact_results_df = regular_season_compact_results_df(access).copy()
    _tourney_season_compact_results_df = tourney_season_compact_results_df(access).copy()

    _regular_season_compact_results_df['Tourney'] = False
    _tourney_season_compact_results_df['Tourney'] = True

    _all_season_compact_results_df = _regular_season_compact_results_df \
        .append(_tourney_season_compact_results_df)
    _all_season_compact_results_df.sort_values(['Season', 'DayNum', 'WTeamID', 'LTeamID'], inplace=True)
    return _all_season_compact_results_df


def _ccompact_results_with_team_names_df(team_names_by_id: Dict[int, str],
                                         compact_results_df: pd.DataFrame) -> pd.DataFrame:
    compact_results_with_team_names_df = compact_results_df.copy()
    compact_results_with_team_names_df['WTeamName'] = compact_results_with_team_names_df.WTeamID \
        .transform(lambda x: team_names_by_id.get(x, ''))
    compact_results_with_team_names_df['LTeamName'] = compact_results_with_team_names_df.LTeamID \
        .transform(lambda x: team_names_by_id.get(x, ''))
    return compact_results_with_team_names_df


def _compact_team_results_df(compact_results_df: pd.DataFrame) -> pd.DataFrame:
    # Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT
    winning_team_results_df = compact_results_df \
        .rename(columns={'WTeamID': 'TeamID',
                         'WTeamName': 'TeamName',
                         'WScore': 'Score',
                         'LTeamID': 'OtherTeamID',
                         'LTeamName': 'OtherTeamName',
                         'LScore': 'OtherScore',
                         'WLoc': 'Loc'})
    losing_team_results_df = compact_results_df \
        .rename(columns={'WTeamID': 'OtherTeamID',
                         'WTeamName': 'OtherTeamName',
                         'WScore': 'OtherScore',
                         'LTeamID': 'TeamID',
                         'LTeamName': 'TeamName',
                         'LScore': 'Score',
                         'WLoc': 'Loc'})
    losing_team_results_df.Loc = losing_team_results_df.Loc.transform(
        lambda x: 'H' if x == 'A' else 'A' if x == 'H' else x)

    _team_results_df = winning_team_results_df.append(losing_team_results_df)

    _team_results_df = _team_results_df[['Season', 'DayNum', 'TeamName', 'TeamID', 'Score',
                                         'OtherTeamName', 'OtherTeamID', 'OtherScore', 'Loc', 'NumOT']]

    _team_results_df.set_index(['TeamID', 'Season', 'DayNum'], inplace=True)
    _team_results_df.sort_index(inplace=True)

    return _team_results_df
