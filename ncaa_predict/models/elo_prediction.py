import pandas as pd

from . import TournamentPredictor


class EloTournamentPredictor(TournamentPredictor):

    def train(self, team_features_df: pd.DataFrame):
        pass

    def estimate_probability(self, season: int, winning_team: int, losing_team: int) -> float:
        return 0.5
