from abc import ABC, abstractmethod
import pandas as pd


class TournamentPredictor(ABC):

    @abstractmethod
    def train(self, team_features_df: pd.DataFrame):
        pass

    @abstractmethod
    def estimate_probability(self, season: int, winning_team: int, losing_team: int) -> float:
        pass
