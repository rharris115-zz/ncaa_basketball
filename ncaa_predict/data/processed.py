from .access import DataAccess
import pandas as pd


def regular_season_compact_team_results_df(access: DataAccess) -> pd.DataFrame:
    return _compact_team_results_df(access=access, compact_results_df=access.regular_season_compact_results_df())


def tourney_season_compact_team_results_df(access: DataAccess) -> pd.DataFrame:
    return _compact_team_results_df(access=access, compact_results_df=access.tourney_compact_results_df())


def _compact_team_results_df(access: DataAccess, compact_results_df: pd.DataFrame) -> pd.DataFrame:
    # Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT
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

    teams = access.teams_df().set_index('TeamID').TeamName.to_dict()

    _team_results_df['TeamName'] = _team_results_df.TeamID.transform(lambda x: teams.get(x, ''))
    _team_results_df['OtherTeamName'] = _team_results_df.OtherTeamID.transform(lambda x: teams.get(x, ''))

    _team_results_df = _team_results_df[['Season', 'DayNum', 'TeamName', 'TeamID', 'Score',
                                         'OtherTeamName', 'OtherTeamID', 'OtherScore', 'Loc', 'NumOT']]

    _team_results_df.set_index(['TeamID', 'Season', 'DayNum'], inplace=True)
    _team_results_df.sort_index(inplace=True)

    return _team_results_df
