from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier

tournament_game_index_labels = ['Season', 'TeamID', 'OtherTeamID']


class TournamentPredictor(ABC):

    @abstractmethod
    def train(self, team_features_df: pd.DataFrame):
        pass

    @abstractmethod
    def estimate_probability(self, tourney_games_df: pd.DataFrame) -> pd.Series:
        pass


class EloTournamentPredictor(TournamentPredictor):

    def train(self, team_features_df: pd.DataFrame):
        self.end_of_regular_season_ratings = team_features_df[~team_features_df.Tourney] \
            .reset_index().groupby(['Season', 'TeamID']).Elo.last()

    def estimate_probability(self, tourney_games_df: pd.DataFrame) -> pd.Series:
        team_elo = tourney_games_df.merge(self.end_of_regular_season_ratings,
                                          on=['Season', 'TeamID'],
                                          how='left').set_index(tournament_game_index_labels).Elo

        other_team_elo = tourney_games_df.merge(self.end_of_regular_season_ratings,
                                                left_on=['Season', 'OtherTeamID'],
                                                right_on=['Season', 'TeamID'],
                                                how='left').set_index(tournament_game_index_labels).Elo

        win_probability = (1 / (1 + 10 ** ((other_team_elo - team_elo) / 400))).rename('Pred')
        return win_probability


class LRTournamentPredictor(TournamentPredictor):

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

    def estimate_probability(self, tourney_games_df: pd.DataFrame) -> pd.Series:
        team_last_features_df = tourney_games_df.merge(self.last_games.drop(columns='OtherTeamID'),
                                                       on=['Season', 'TeamID'],
                                                       how='left').set_index(tournament_game_index_labels)

        other_team_last_features_df = tourney_games_df.merge(self.last_games.drop(columns='OtherTeamID'),
                                                             left_on=['Season', 'OtherTeamID'],
                                                             right_on=['Season', 'TeamID'],
                                                             how='left').set_index(tournament_game_index_labels)

        to_drop = ['DayNum', 'Tourney', 'Win', 'RestDaysMax7']
        team_last_features_df = team_last_features_df.drop(columns=to_drop)
        other_team_last_features_df = other_team_last_features_df.drop(columns=to_drop)

        team_last_features_df.rename(columns=lambda c: f'p_{c}', inplace=True)
        other_team_last_features_df.rename(columns=lambda c: f'po_{c}', inplace=True)

        x = team_last_features_df.join(other_team_last_features_df)
        x['HomeAdvantage'] = 0
        x['p_EloAdv'] = x.p_Elo - x.po_Elo
        x = x.drop(columns=['p_Elo', 'po_Elo'])
        x = x.reindex(sorted(x.columns), axis=1)

        p = self.lr.predict_proba(x)
        return pd.Series(index=x.index, name='Pred', data=p[:, 1])


class MLPTournamentPredictor(TournamentPredictor):

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

        self.mlp = MLPClassifier(hidden_layer_sizes=(35, 35), verbose=True, max_iter=1000).fit(x, y)

    def estimate_probability(self, tourney_games_df: pd.DataFrame) -> pd.Series:
        team_last_features_df = tourney_games_df.merge(self.last_games.drop(columns='OtherTeamID'),
                                                       on=['Season', 'TeamID'],
                                                       how='left').set_index(tournament_game_index_labels)

        other_team_last_features_df = tourney_games_df.merge(self.last_games.drop(columns='OtherTeamID'),
                                                             left_on=['Season', 'OtherTeamID'],
                                                             right_on=['Season', 'TeamID'],
                                                             how='left').set_index(tournament_game_index_labels)

        to_drop = ['DayNum', 'Tourney', 'Win', 'RestDaysMax7']
        team_last_features_df = team_last_features_df.drop(columns=to_drop)
        other_team_last_features_df = other_team_last_features_df.drop(columns=to_drop)

        team_last_features_df.rename(columns=lambda c: f'p_{c}', inplace=True)
        other_team_last_features_df.rename(columns=lambda c: f'po_{c}', inplace=True)

        x = team_last_features_df.join(other_team_last_features_df)
        x['HomeAdvantage'] = 0
        x['p_EloAdv'] = x.p_Elo - x.po_Elo
        x = x.drop(columns=['p_Elo', 'po_Elo'])
        x = x.reindex(sorted(x.columns), axis=1)

        p = self.mlp.predict_proba(x)
        return pd.Series(index=x.index, name='Pred', data=p[:, 1])


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
