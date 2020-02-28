from ncaa_predict.data.access import mens_access, womens_access
from ncaa_predict.features import tf
from ncaa_predict.models.elo_prediction import EloTournamentPredictor
from ncaa_predict.data.processed import possible_games
from ncaa_predict.evaluate import log_loss_error
import pandas as pd
from tqdm import tqdm


def main():
    for access in (womens_access, mens_access):
        team_features_df = pd.read_pickle(f'{access.prefix}TeamFeatures.pkl')

        pred = EloTournamentPredictor()
        pred.train(team_features_df=team_features_df)

        tourney_games = [(season, ta, tb)
                         for season, ta, tb in possible_games(access)
                         if season >= 2015]

        predictions_df = pd.DataFrame.from_records([
            {
                'ID': f'{season}_{ta}_{tb}',
                'Pred': pred.estimate_probability(season=season, winning_team=ta, losing_team=tb)
            }
            for season, ta, tb in tqdm(iterable=tourney_games, desc='Recording predictions')
        ]).set_index('ID').sort_index()

        print(f'Log Loss: {log_loss_error(predictions_df=predictions_df, access=access)}')

        predictions_df.to_csv(f'{access.prefix}SubmissionStage1.csv', index=True)


if __name__ == '__main__':
    main()
