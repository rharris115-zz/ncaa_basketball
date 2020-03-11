from collections import defaultdict
from typing import Iterable, Tuple

import pandas as pd

from .access import DataAccess
from ..utils import memoize

team_format_indices = ['TeamID', 'Season', 'DayNum', 'OtherTeamID']
game_format_indices = ['Season', 'DayNum', 'WTeamID', 'LTeamID']
player_game_format_indices = game_format_indices + ['EventPlayerID']


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


def possible_games(access: DataAccess) -> Iterable[Tuple[int, int, int]]:
    # Season, Seed, TeamID
    seeds_df = access.tourney_seeds_df()
    for season, season_seeds_df in seeds_df.groupby('Season'):
        teams = season_seeds_df.TeamID.unique().tolist()
        for ta in teams:
            for tb in teams:
                if ta < tb:
                    yield season, ta, tb


def slot_paths_df(access: DataAccess):
    slots = access.tourney_slots_df()

    def _paths(season_slots_df: pd.DataFrame):
        seed_segments = {**{row.StrongSeed: row.Slot for idx, row in season_slots_df.iterrows()},
                         **{row.WeakSeed: row.Slot for idx, row in season_slots_df.iterrows()}}

        initial_seeds = sorted({*seed_segments.keys()} - {*seed_segments.values()})

        def _find_slots(slot_or_seed: str) -> Iterable[str]:
            if slot_or_seed in seed_segments:
                slot = seed_segments[slot_or_seed]
                yield slot
                yield from _find_slots(slot_or_seed=slot)

        paths = {seed: list(_find_slots(slot_or_seed=seed)) for seed in sorted(initial_seeds)}

        season_paths_df = pd.DataFrame.from_records(({'Seed': seed, 'path': path}
                                                     for seed, path in paths.items()),
                                                    index='Seed')

        return season_paths_df

    paths_df = slots.groupby('Season').apply(_paths).reset_index()

    return paths_df


@memoize
def paths_to_championship_df(access: DataAccess) -> pd.DataFrame:
    paths_df = slot_paths_df(access=access)
    seeds = access.tourney_seeds_df()
    paths_2_championship_df = paths_df.merge(seeds, on=['Season', 'Seed'], how='outer').set_index(['Season', 'TeamID'])
    return paths_2_championship_df


@memoize
def infer_slot_dates(access: DataAccess):
    tourney_compact_results_df = access.tourney_compact_results_df()[['Season', 'DayNum', 'WTeamID', 'LTeamID']].copy()
    paths_by_season_and_team = paths_to_championship_df(access=access).path.to_dict()

    def _slot(row: pd.Series):
        w_path = paths_by_season_and_team[(row.Season, row.WTeamID)]
        l_path = paths_by_season_and_team[(row.Season, row.LTeamID)]
        for slot in w_path:
            if slot in l_path:
                return slot
        return None

    tourney_compact_results_df['Slot'] = tourney_compact_results_df.apply(_slot, axis=1)
    slot_dates_df = tourney_compact_results_df.drop(columns=['WTeamID', 'LTeamID'])
    return slot_dates_df


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
