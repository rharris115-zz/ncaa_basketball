from . import registry
from ..data.processed import to_team_format
from ..data.access import DataAccess
import pandas as pd
from ..data.processed import all_season_compact_results_df
from typing import Dict
from tqdm import tqdm

r_0 = 1300


@registry.register
def elo(access: DataAccess) -> pd.Series:
    # https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
    # https://www.ergosum.co/nate-silvers-nba-elo-algorithm/

    _all_season_compact_results_df = all_season_compact_results_df(access).reset_index()

    w_elo, l_elo = [], []
    team_elo = {}  # type: Dict[int,float]

    team_conferences_df = access.team_conferences_df()

    teams_by_season_and_conference = team_conferences_df.groupby('Season') \
        .apply(lambda x: x.groupby('ConfAbbrev').TeamID.apply(list).to_dict()).to_dict()

    for season, season_df in tqdm(iterable=_all_season_compact_results_df.groupby('Season'), leave=False,
                                  desc='Calculating Historical Elo Ratings'):

        # We apply a regression to conference mean at the start of each season.
        teams_by_conference = teams_by_season_and_conference.get(season, {})
        conference_average_elo = {conference: sum(team_elo.get(team, r_0) for team in teams) / len(teams)
                                  for conference, teams in tqdm(iterable=teams_by_conference.items(), leave=False,
                                                                desc=f'Calculating conference means at start of season {season}')}

        for conference, teams in tqdm(iterable=teams_by_conference.items(), leave=False,
                                      desc=f'Applying conference mean regression at start of season {season}'):
            average_elo = conference_average_elo.get(conference, r_0)
            for team in teams:
                team_elo[team] = 0.25 * average_elo + 0.75 * team_elo.get(team, r_0)

        # Apply Elo to season games.
        for i, row in tqdm(iterable=season_df.iterrows(),
                           desc=f'Calculating Elo rankings for season {season}',
                           leave=False):
            w_loc = row.WLoc
            r_w = team_elo.get(row.WTeamID, r_0)
            r_l = team_elo.get(row.LTeamID, r_0)

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
