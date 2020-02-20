from . import registry
from ..data.access import DataAccess
import pandas as pd
from ..data.processed import all_season_compact_results_df
from typing import Dict

r_0 = 1300


@registry.register
def elo(access: DataAccess) -> pd.DataFrame:
    # https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
    # https://www.ergosum.co/nate-silvers-nba-elo-algorithm/

    _all_season_compact_results_df = all_season_compact_results_df(access).reset_index()
    w_elo, l_elo = [], []
    team_elo = {}  # type: Dict[int,float]

    team_conferences_df = access.team_conferences_df()

    teams_by_season_and_conference = team_conferences_df.groupby('Season') \
        .apply(lambda x: x.groupby('ConfAbbrev').TeamID.apply(list).to_dict()).to_dict()

    for season, season_df in _all_season_compact_results_df.groupby('Season'):

        # We apply a regression to conference mean at the start of each season.
        teams_by_conference = teams_by_season_and_conference.get(season, {})
        conference_average_elo = {conference: sum(team_elo.get(team, r_0) for team in teams) / len(teams)
                                  for conference, teams in teams_by_conference.items()}

        for conference, teams in teams_by_conference.items():
            average_elo = conference_average_elo.get(conference, r_0)
            for team in teams:
                team_elo[team] = 0.25 * average_elo + 0.75 * team_elo.get(team, r_0)

        # Apply Elo to season games.
        for i, row in season_df.iterrows():
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

    _all_season_compact_results_df['WElo'] = w_elo
    _all_season_compact_results_df['LElo'] = l_elo

    _all_season_compact_results_df.set_index(['Season', 'DayNum', 'WTeamID', 'LTeamID'], inplace=True)
    _all_season_compact_results_df.sort_index(inplace=True)

    return _all_season_compact_results_df[['WElo', 'LElo']]
