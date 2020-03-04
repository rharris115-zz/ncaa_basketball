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
    def estimate_probability(self, season: int, team_a: int, team_b: int) -> float:
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

    def estimate_probability(self, season: int, team_a: int, team_b: int) -> float:
        r_w = self.end_of_season_ratings[season][team_a]
        r_l = self.end_of_season_ratings[season][team_b]
        win_probability = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        return win_probability


class LRTournamentPredictor(TournamentPredictor):

    def __init__(self):
        pass

    def train(self, team_features_df: pd.DataFrame):
        self.last_games = team_features_df[~team_features_df.Tourney].reset_index().groupby(['TeamID', 'Season']).last()

        x, y = training_data_df(team_features_df=team_features_df)

        x['p_EloAdv'] = x.p_Elo - x.po_Elo
        # x['RestDaysAdv'] = x.RestDaysMax7 - x.OtherRestDaysMax7

        x.drop(columns=['p_Win', 'po_Win', 'p_Elo', 'po_Elo', 'p_RestDaysMax7', 'po_RestDaysMax7',
                        'RestDaysMax7', 'OtherRestDaysMax7', 'Tourney', 'p_Tourney', 'po_Tourney'], inplace=True)

        x = x.reindex(sorted(x.columns), axis=1)

        x = x[x.index.get_level_values('TeamID') < x.index.get_level_values('OtherTeamID')]
        y = y[y.index.get_level_values('TeamID') < y.index.get_level_values('OtherTeamID')]

        self.lr = LogisticRegression(random_state=0, max_iter=1e6).fit(x, y)

    def estimate_probability(self, season: int, team_a: int, team_b: int) -> float:
        team_a_last_result = self.last_games.loc[(team_a, season)]
        team_b_last_result = self.last_games.loc[(team_b, season)]

        # 'p_Score', 'p_NumOT', 'p_ScoreDifference', 'p_Streak', 'p_HomeAdvantage', 'p_AssistEntropy', 'p_ScoringEntropy'

        to_drop = ['OtherTeamID', 'DayNum', 'Tourney', 'Win', 'RestDaysMax7']
        team_a_last_result = team_a_last_result.drop(to_drop)
        team_b_last_result = team_b_last_result.drop(to_drop)

        a_x = team_a_last_result.rename({name: f'p_{name}' for name in team_a_last_result.index})
        b_x = team_b_last_result.rename({name: f'po_{name}' for name in team_a_last_result.index})

        x = a_x.append(b_x)
        x['HomeAdvantage'] = 0
        x['p_EloAdv'] = x.p_Elo - x.po_Elo
        x = x.drop(['p_Elo', 'po_Elo'])
        x = x.to_frame().transpose()
        x = x.reindex(sorted(x.columns), axis=1)

        p = self.lr.predict_proba(x)
        # print(f'p_Elo: {team_a_last_result.Elo}, po_Elo: {team_b_last_result.Elo}, p: {p[0, 1]}')
        return p[0, 1]


def training_data_df(team_features_df: pd.DataFrame):
    tf_df = team_features_df.reset_index()
    tf_df.set_index(['Season', 'DayNum', 'TeamID', 'OtherTeamID'], inplace=True)
    tf_df.sort_index(inplace=True)
    tf_df = tf_df[['Win', 'HomeAdvantage', 'Tourney', 'RestDaysMax7']]

    other_known_df = team_features_df.RestDaysMax7.rename('OtherRestDaysMax7').reset_index() \
        .rename(columns=lambda x: 'OtherTeamID' if x == 'TeamID' else 'TeamID' if x == 'OtherTeamID' else x)
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
