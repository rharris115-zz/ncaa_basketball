from .access import DataAccess
from ..utils import memoize
from typing import Dict, Iterable, Tuple, List, Any, Sequence
import pandas as pd
import numpy as np

team_format_indices = ['TeamID', 'Season', 'DayNum', 'OtherTeamID']
game_format_indices = ['Season', 'DayNum', 'WTeamID', 'LTeamID']
player_game_format_indices = game_format_indices + ['EventPlayerID']


@memoize
def regular_season_compact_team_results_df(access: DataAccess) -> pd.DataFrame:
    compact_results_df = regular_season_compact_results_df(access)
    return _compact_team_results_df(compact_results_with_team_names_df=compact_results_df)


@memoize
def tourney_season_compact_team_results_df(access: DataAccess) -> pd.DataFrame:
    compact_results_df = tourney_season_compact_results_df(access)
    return _compact_team_results_df(compact_results_with_team_names_df=compact_results_df)


@memoize
def all_compact_team_results_df(access: DataAccess) -> pd.DataFrame:
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
def ground_truth_since_2015(access: DataAccess) -> pd.DataFrame:
    compact_results_df = access.tourney_compact_results_df()

    compact_results_df = compact_results_df[compact_results_df.Season >= 2015]

    smaller_team_id_wins = compact_results_df[compact_results_df.WTeamID < compact_results_df.LTeamID]
    larger_team_id_wins = compact_results_df[compact_results_df.WTeamID > compact_results_df.LTeamID]

    _s_ground_truth_df = smaller_team_id_wins.Season.astype(str).str \
        .cat([smaller_team_id_wins.WTeamID.astype(str), smaller_team_id_wins.LTeamID.astype(str)], sep='_') \
        .rename('ID').to_frame()
    _g_ground_truth_df = larger_team_id_wins.Season.astype(str).str \
        .cat([larger_team_id_wins.LTeamID.astype(str), larger_team_id_wins.WTeamID.astype(str)], sep='_') \
        .rename('ID').to_frame()

    _s_ground_truth_df['Win'] = 1
    _g_ground_truth_df['Win'] = 0

    _ground_truth_df = _s_ground_truth_df.append(_g_ground_truth_df).set_index('ID')
    _ground_truth_df.sort_index(inplace=True)

    return _ground_truth_df


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

    compact_results_with_team_names_df.set_index(game_format_indices, inplace=True)
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

    team_formatted_df.set_index(team_format_indices, inplace=True)
    team_formatted_df.sort_index(inplace=True)

    return team_formatted_df


def possible_games(access: DataAccess) -> Iterable[Tuple[int, int, int]]:
    # Season, Seed, TeamID
    seeds_df = access.tourney_seeds_df()
    for season, season_seeds_df in seeds_df.groupby('Season'):
        teams = season_seeds_df.TeamID.unique().tolist()
        for ta in teams:
            for tb in teams:
                if ta < tb:
                    yield season, ta, tb


@memoize
def filtered_events_df(*event_types: Sequence[str], access: DataAccess, season: int) -> pd.DataFrame:
    events_df = access.events_df(season=season)
    filtered_df = events_df[events_df.EventType.isin({*event_types})]
    return filtered_df


def player_scoring_df(access: DataAccess) -> pd.DataFrame:
    # EventID, Season, DayNum, WTeamID, LTeamID, WFinalScore, LFinalScore, WCurrentScore, LCurrentScore,
    # ElapsedSeconds, EventTeamID, EventPlayerID, EventType, EventSubType, X, Y, Area
    p_scoring_df = None
    for season in range(2015, 2020):
        filtered_events = filtered_events_df('made1', 'made2', 'made3', 'miss1', 'miss2', 'miss3',
                                             access=access, season=season)
        season_p_scoring_df = filtered_events.pivot_table(index=player_game_format_indices,
                                                          columns='EventType',
                                                          values='EventID',
                                                          aggfunc=np.count_nonzero).fillna(0)
        p_scoring_df = season_p_scoring_df \
            if p_scoring_df is None \
            else p_scoring_df.append(season_p_scoring_df)

    return p_scoring_df
