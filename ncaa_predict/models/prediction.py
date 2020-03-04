from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import numpy as np


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
        x['p_EloAdv'] = x.p_Elo - x.po_Elo
        x['RestDaysAdv'] = x.RestDaysMax7 - x.OtherRestDaysMax7
        x.drop(columns=['p_Win', 'po_Win', 'p_Elo', 'po_Elo', 'p_RestDaysMax7', 'po_RestDaysMax7',
                        'RestDaysMax7', 'OtherRestDaysMax7', 'Tourney', 'p_Tourney', 'po_Tourney'], inplace=True)
        lr = LogisticRegression(random_state=0, max_iter=1e6).fit(x, y)
        for col, coef in zip(x.columns, np.nditer(lr.coef_)):
            print(col, coef)
        pass

    def estimate_probability(self, season: int, winning_team: int, losing_team: int) -> float:
        return 0.5


def training_data_df(team_features_df: pd.DataFrame):
    tf_df = team_features_df.reset_index()
    tf_df = tf_df[tf_df.TeamID < tf_df.OtherTeamID]
    tf_df.set_index(['Season', 'DayNum', 'TeamID', 'OtherTeamID'], inplace=True)
    tf_df.sort_index(inplace=True)
    tf_df = tf_df[['Win', 'HomeAdvantage', 'Tourney', 'RestDaysMax7']]

    other_known_df = team_features_df.RestDaysMax7.rename('OtherRestDaysMax7').reset_index() \
        .rename(columns=lambda x: 'OtherTeamID' if x == 'TeamID' else 'TeamID' if x == 'OtherTeamID' else x)
    other_known_df = other_known_df[other_known_df.TeamID < other_known_df.OtherTeamID]
    other_known_df.set_index(['Season', 'DayNum', 'TeamID', 'OtherTeamID'], inplace=True)
    other_known_df.sort_index(inplace=True)

    tf_df = tf_df.join(other_known_df)
    tf_df.reset_index(inplace=True)

    previous_team_attributes_df = team_features_df.reset_index().set_index(['TeamID', 'Season', 'DayNum']) \
        .groupby(by='TeamID').shift(1)

    p_attributes_df = previous_team_attributes_df.drop(columns='OtherTeamID') \
        .rename(columns=lambda c: 'p_' + c).reset_index()
    po_team_attributes_df = previous_team_attributes_df.drop(columns='OtherTeamID') \
        .rename(columns=lambda c: 'po_' + c).reset_index() \
        .rename(columns={'TeamID': 'OtherTeamID'})

    data_df = pd.merge(left=tf_df, right=p_attributes_df,
                       left_on=['Season', 'DayNum', 'TeamID'], right_on=['Season', 'DayNum', 'TeamID'])
    data_df = pd.merge(left=data_df, right=po_team_attributes_df,
                       left_on=['Season', 'DayNum', 'OtherTeamID'], right_on=['Season', 'DayNum', 'OtherTeamID'])

    data_df.set_index(['Season', 'DayNum', 'TeamID', 'OtherTeamID'], inplace=True)
    data_df.sort_index(inplace=True)
    data_df.dropna(inplace=True)

    return data_df.drop(columns='Win'), data_df.Win
