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

    _all_season_compact_results_df = all_season_compact_results_df(access).copy()
    w_elo, l_elo, w_elo_after, l_elo_after = [], [], [], []
    team_elo = {}  # type: Dict[int,float]

    team_conferences_df = access.team_conferences_df()

    def _f(x):
        return x

    team_conferences_df.groupby('ConfAbbrev').apply(_f)

    for season, season_df in _all_season_compact_results_df.groupby('Season'):

        for i, row in season_df.iterrows():
            w_loc = row.WLoc
            r_w = team_elo.get(row.WTeamID, r_0) + (100 if w_loc == 'H' else 0)
            r_l = team_elo.get(row.LTeamID, r_0) + (100 if w_loc == 'A' else 0)

            elo_diff_w = r_w - r_l
            mov_w = row.WScore - row.LScore

            k = 20 * ((mov_w + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_w)

            s_w = 1
            s_l = 0

            e_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
            e_l = 1 / (1 + 10 ** ((r_w - r_l) / 400))

            r_w_new = k * (s_w - e_w) + r_w
            r_l_new = k * (s_l - e_l) + r_l

            team_elo[row.WTeamID] = r_w_new
            team_elo[row.LTeamID] = r_l_new

            w_elo.append(r_w)
            l_elo.append(r_l)
            w_elo_after.append(r_w_new)
            l_elo_after.append(r_l_new)

    _all_season_compact_results_df['WElo'] = w_elo
    _all_season_compact_results_df['LElo'] = l_elo
    _all_season_compact_results_df['WEloAfter'] = w_elo_after
    _all_season_compact_results_df['LEloAfter'] = l_elo_after

    return pd.DataFrame()
