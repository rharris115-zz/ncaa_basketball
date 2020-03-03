from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from ..data.processed import game_format_indices


class TournamentPredictor(ABC):

    @abstractmethod
    def train(self, team_features_df: pd.DataFrame):
        pass

    @abstractmethod
    def estimate_probability(self, season: int, winning_team: int, losing_team: int) -> float:
        pass


class EloTournamentPredictor(TournamentPredictor):

    def __init__(self):
        self.end_of_season_ratings = defaultdict(dict)

    def train(self, team_features_df: pd.DataFrame):
        for (season, team_id), season_features_df in tqdm(
                iterable=team_features_df[team_features_df.Tourney == False].groupby(['Season', 'TeamID']),
                desc='Extracting end of regular season Elo ratings'
        ):
            self.end_of_season_ratings[season][team_id] = season_features_df.Elo[-1]

    def estimate_probability(self, season: int, winning_team: int, losing_team: int) -> float:
        r_w = self.end_of_season_ratings[season][winning_team]
        r_l = self.end_of_season_ratings[season][losing_team]
        win_probability = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        return win_probability


class LRTournamentPredictor(TournamentPredictor):

    def __init__(self):
        pass

    def train(self, team_features_df: pd.DataFrame):
        x, y = training_data_df(team_features_df=team_features_df)
        pass

    def estimate_probability(self, season: int, winning_team: int, losing_team: int) -> float:
        return 0.5


def training_data_df(team_features_df: pd.DataFrame):
    tf_df = team_features_df.reset_index()
    tf_df = tf_df[tf_df.TeamID < tf_df.OtherTeamID]
    tf_df.set_index(['Season', 'DayNum', 'TeamID', 'OtherTeamID'], inplace=True)
    tf_df.sort_index(inplace=True)

    tf_df = tf_df[['Win', 'HomeAdvantage', 'Tourney', 'RestDaysMax14', 'RestDaysMax7', 'RestDaysMax3']].reset_index()

    previous_team_attributes_df = team_features_df.reset_index().set_index(['TeamID', 'Season', 'DayNum']) \
        .groupby(by='TeamID').shift(1)

    p_attributes_df = previous_team_attributes_df.rename(columns=lambda c: 'p_' + c).reset_index()
    po_team_attributes_df = previous_team_attributes_df.rename(columns=lambda c: 'po_' + c).reset_index() \
        .rename(columns={'TeamID': 'OtherTeamID'})

    data_df = pd.merge(left=tf_df, right=p_attributes_df,
                       left_on=['Season', 'DayNum', 'TeamID'], right_on=['Season', 'DayNum', 'TeamID'])
    data_df = pd.merge(left=data_df, right=po_team_attributes_df,
                       left_on=['Season', 'DayNum', 'OtherTeamID'], right_on=['Season', 'DayNum', 'OtherTeamID'])

    data_df.set_index(['Season', 'DayNum', 'TeamID', 'OtherTeamID'], inplace=True)
    data_df.sort_index(inplace=True)
    data_df.dropna(inplace=True)

    return data_df.drop(columns='Win'), data_df.Win
