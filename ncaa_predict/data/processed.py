from typing import Iterable, Tuple

import pandas as pd

from .access import DataAccess

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
