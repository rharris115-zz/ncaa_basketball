from .access import DataAccess
import pandas as pd


def team_results_df(access: DataAccess) -> pd.DataFrame:
    # Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT
    regular_season_compact_results_df = access.regular_season_compact_results_df()
    tourney_compact_results_df = access.tourney_compact_results_df()

    regular_season_compact_results_df['playoff'] = False
    tourney_compact_results_df['playoff'] = True

    compact_results_df = regular_season_compact_results_df \
        .append(tourney_compact_results_df)

    winning_team_results_df = compact_results_df \
        .rename(columns={'WTeamID': 'TeamID',
                         'WScore': 'Score',
                         'LTeamID': 'OtherTeamID',
                         'LScore': 'OtherScore',
                         'WLoc': 'Loc'})
    losing_team_results_df = compact_results_df \
        .rename(columns={'WTeamID': 'OtherTeamID',
                         'WScore': 'OtherScore',
                         'LTeamID': 'TeamID',
                         'LScore': 'Score',
                         'WLoc': 'Loc'})
    losing_team_results_df.Loc = losing_team_results_df.Loc.transform(
        lambda x: 'H' if x == 'A' else 'A' if x == 'H' else x)

    _team_results_df = winning_team_results_df.append(losing_team_results_df)

    _team_results_df.set_index(['TeamID', 'Season', 'DayNum'], inplace=True)
    _team_results_df.sort_index(inplace=True)

    return _team_results_df
