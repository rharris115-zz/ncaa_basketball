from typing import Dict

import pandas as pd
from tqdm import tqdm

from . import tf
from ..data.access import DataAccess
from ..data.processed import team_format_indices, game_format_indices, to_team_format
from ..utils import memoize


@tf.register
def is_tourney(access: DataAccess) -> pd.Series:
    ctr_df = all_compact_team_results_df(access)
    return ctr_df.Tourney.rename('Tourney')


@tf.register
def score_difference(access: DataAccess) -> pd.Series:
    ctr_df = all_compact_team_results_df(access)
    return (ctr_df.Score - ctr_df.OtherScore).rename('ScoreDifference')


@tf.register
def win(access: DataAccess) -> pd.Series:
    ctr_df = all_compact_team_results_df(access)
    return (((ctr_df.Score > ctr_df.OtherScore).astype(int) * 2) - 1).rename('Win')


@tf.register
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


@tf.register
def home_advantage(access: DataAccess) -> pd.Series:
    ctr_df = all_compact_team_results_df(access).copy()
    ctr_df.loc[ctr_df.Loc == 'H', 'HomeAdvantage'] = 1
    ctr_df.loc[ctr_df.Loc == 'N', 'HomeAdvantage'] = 0
    ctr_df.loc[ctr_df.Loc == 'A', 'HomeAdvantage'] = -1
    return ctr_df.HomeAdvantage


@tf.register
@memoize
def rest_days(access: DataAccess) -> pd.Series:
    ctr_df = all_compact_team_results_df(access).reset_index()
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


@tf.register
def rest_days_14_max(access: DataAccess):
    return _rest_days_with_maximum(access=access, maximum=14)


@tf.register
def rest_days_7_max(access: DataAccess):
    return _rest_days_with_maximum(access=access, maximum=7)


@tf.register
def rest_days_3_max(access: DataAccess):
    return _rest_days_with_maximum(access=access, maximum=3)


elo_r_0 = 1300


@tf.register
def elo(access: DataAccess) -> pd.Series:
    # https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
    # https://www.ergosum.co/nate-silvers-nba-elo-algorithm/

    _all_season_compact_results_df = all_season_compact_results_df(access).reset_index()

    w_elo, l_elo = [], []
    team_elo = {}  # type: Dict[int,float]

    team_conferences_df = access.team_conferences_df()

    teams_by_season_and_conference = team_conferences_df.groupby('Season') \
        .apply(lambda x: x.groupby('ConfAbbrev').TeamID.apply(list).to_dict()).to_dict()

    for season, season_df in tqdm(iterable=_all_season_compact_results_df.groupby('Season'),
                                  leave=False, total=_all_season_compact_results_df.Season.nunique(),
                                  desc='Calculating Historical Elo Ratings'):

        # We apply a regression to conference mean at the start of each season.
        teams_by_conference = teams_by_season_and_conference.get(season, {})
        conference_average_elo = {conference: sum(team_elo.get(team, elo_r_0) for team in teams) / len(teams)
                                  for conference, teams in tqdm(iterable=teams_by_conference.items(), leave=False,
                                                                desc=f'Calculating conference means at start of season {season}')}

        for conference, teams in tqdm(iterable=teams_by_conference.items(), leave=False,
                                      desc=f'Applying conference mean regression at start of season {season}'):
            average_elo = conference_average_elo.get(conference, elo_r_0)
            for team in teams:
                team_elo[team] = 0.25 * average_elo + 0.75 * team_elo.get(team, elo_r_0)

        # Apply Elo to season games.
        for i, row in tqdm(iterable=season_df.iterrows(),
                           desc=f'Calculating Elo rankings for season {season}',
                           leave=False, total=len(season_df.index)):
            w_loc = row.WLoc
            r_w = team_elo.get(row.WTeamID, elo_r_0)
            r_l = team_elo.get(row.LTeamID, elo_r_0)

            r_w_ha = r_w + (100 if w_loc == 'H' else 0)
            r_l_ha = r_l + (100 if w_loc == 'A' else 0)

            elo_diff_w = r_w_ha - r_l_ha
            mov_w = row.WScore - row.LScore

            k = 20 * ((mov_w + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_w)

            s_w = 1
            s_l = 0

            e_w = 1 / (1 + 10 ** ((r_l_ha - r_w_ha) / 400))
            e_l = 1 / (1 + 10 ** ((r_w_ha - r_l_ha) / 400))

            r_w_new = k * (s_w - e_w) + r_w
            r_l_new = k * (s_l - e_l) + r_l

            team_elo[row.WTeamID] = r_w_new
            team_elo[row.LTeamID] = r_l_new

            w_elo.append(r_w_new)
            l_elo.append(r_l_new)

    _elo_game_formatted_df = _all_season_compact_results_df[['Season', 'DayNum', 'WTeamID', 'LTeamID']].copy()
    _elo_game_formatted_df['WElo'] = w_elo
    _elo_game_formatted_df['LElo'] = l_elo

    _elo_df = to_team_format(game_formatted_df=_elo_game_formatted_df)

    return _elo_df.Elo


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
