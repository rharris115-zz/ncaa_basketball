from .access import DataAccess
from ..utils import memoize
from typing import Dict
import pandas as pd


@memoize
def regular_season_compact_team_results_df(access: DataAccess) -> pd.DataFrame:
    compact_results_df = regular_season_compact_results_df(access)
    return _compact_team_results_df(compact_results_with_team_names_df=compact_results_df)


@memoize
def tourney_season_compact_team_results_df(access: DataAccess) -> pd.DataFrame:
    compact_results_df = tourney_season_compact_results_df(access)
    return _compact_team_results_df(compact_results_with_team_names_df=compact_results_df)


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
    return _compact_results_with_team_names_df(team_names_by_id=team_names_by_id,
                                               compact_results_df=compact_results_df)


@memoize
def tourney_season_compact_results_df(access: DataAccess) -> pd.DataFrame:
    compact_results_df = access.tourney_compact_results_df()
    team_names_by_id = access.teams_df().set_index('TeamID').TeamName.to_dict()
    return _compact_results_with_team_names_df(team_names_by_id=team_names_by_id,
                                               compact_results_df=compact_results_df)


@memoize
def all_season_compact_results_df(access: DataAccess) -> pd.DataFrame:
    _regular_season_compact_results_df = regular_season_compact_results_df(access).copy()
    _tourney_season_compact_results_df = tourney_season_compact_results_df(access).copy()

    _regular_season_compact_results_df['Tourney'] = False
    _tourney_season_compact_results_df['Tourney'] = True

    _all_season_compact_results_df = _regular_season_compact_results_df \
        .append(_tourney_season_compact_results_df)
    _all_season_compact_results_df.sort_index(inplace=True)
    return _all_season_compact_results_df


def _compact_results_with_team_names_df(team_names_by_id: Dict[int, str],
                                        compact_results_df: pd.DataFrame) -> pd.DataFrame:
    compact_results_with_team_names_df = compact_results_df.copy()
    compact_results_with_team_names_df['WTeamName'] = compact_results_with_team_names_df.WTeamID \
        .transform(lambda x: team_names_by_id.get(x, ''))
    compact_results_with_team_names_df['LTeamName'] = compact_results_with_team_names_df.LTeamID \
        .transform(lambda x: team_names_by_id.get(x, ''))

    compact_results_with_team_names_df.set_index(['Season', 'DayNum', 'WTeamID', 'LTeamID'], inplace=True)
    compact_results_with_team_names_df.sort_index(inplace=True)
    return compact_results_with_team_names_df


def _compact_team_results_df(compact_results_with_team_names_df: pd.DataFrame) -> pd.DataFrame:
    # Season, DayNum, WTeamID, WTeamName, WScore, LTeamID, LTeamName, LScore, WLoc, NumOT
    return to_team_format(game_formatted_df=compact_results_with_team_names_df)


def _winning_column_renamer(column: str) -> str:
    if column.startswith('W'):
        return column[1:]
    elif column.startswith('L'):
        return 'Other' + column[1:]
    else:
        return column


def _losing_column_ranamer(column: str) -> str:
    if column.startswith('L'):
        return column[1:]
    elif column.startswith('W'):
        return 'Other' + column[1:]
    else:
        return column


def to_team_format(game_formatted_df: pd.DataFrame) -> pd.DataFrame:
    winning_game_formatted_df = game_formatted_df.reset_index()
    winning_game_formatted_df.rename(columns=_winning_column_renamer, inplace=True)

    losing_game_formatted_df = game_formatted_df.reset_index()
    losing_game_formatted_df.rename(columns=_losing_column_ranamer, inplace=True)

    if 'OtherLoc' in losing_game_formatted_df.columns:
        losing_game_formatted_df['Loc'] = losing_game_formatted_df.OtherLoc \
            .transform(lambda x: 'H' if x == 'A' else 'A' if x == 'H' else x)
        losing_game_formatted_df.drop(columns='OtherLoc', inplace=True)

    team_formatted_df = winning_game_formatted_df.append(losing_game_formatted_df)

    team_formatted_df.set_index(['TeamID', 'Season', 'DayNum'], inplace=True)
    team_formatted_df.sort_index(inplace=True)

    return team_formatted_df
