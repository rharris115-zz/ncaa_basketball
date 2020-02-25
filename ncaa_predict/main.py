from ncaa_predict.data.access import mens_access, womens_access
from ncaa_predict.features import registry
import pandas as pd
from ncaa_predict.models.elo_prediction import EloTournamentPredictor
from ncaa_predict.data.processed import possible_games


def main():
    d = registry.run(access=womens_access)
    team_features_df = pd.concat(d.values(), axis=1)

    pred = EloTournamentPredictor()
    pred.train(team_features_df=team_features_df)

    predictions = pd.DataFrame.from_records([
        {
            'Season': season,
            'WinningTeam': ta,
            'LosingTeam': tb,
            'Pred': pred.estimate_probability(season=season, winning_team=ta, losing_team=tb)
        }
        for season, ta, tb in possible_games(womens_access)
    ])

    print(predictions)


if __name__ == '__main__':
    main()
