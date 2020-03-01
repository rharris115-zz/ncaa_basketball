import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from abc import ABC, abstractmethod
import pandas as pd


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
        training_data_df(team_features_df=team_features_df)

    def estimate_probability(self, season: int, winning_team: int, losing_team: int) -> float:
        return 0.5


def training_data_df(team_features_df: pd.DataFrame):
    pass
