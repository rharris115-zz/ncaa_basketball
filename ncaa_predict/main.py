import pandas as pd
from tqdm import tqdm

from ncaa_predict.data.access import mens_access, womens_access, team_features_df, team_player_features
from ncaa_predict.data.processed import possible_games, team_format_indices
from ncaa_predict.evaluate import log_loss_error
from ncaa_predict.models.prediction import EloTournamentPredictor, LRTournamentPredictor


def main():
    for access in (mens_access, womens_access):
        tpf_df = team_player_features(prefix=access.prefix)
        tf_df = team_features_df(prefix=access.prefix)

        tf_df = tf_df.join(tpf_df, how='right')

        # elo_pred = EloTournamentPredictor()
        # elo_pred.train(team_features_df=tf_df)

        lr_pred = LRTournamentPredictor()
        lr_pred.train(team_features_df=tf_df)

        tourney_games = [(season, ta, tb)
                         for season, ta, tb in possible_games(access)
                         if season >= 2015]

        predictions_df = pd.DataFrame.from_records(
            {
                'ID': f'{season}_{ta}_{tb}',
                'Pred': elo_pred.estimate_probability(season=season, winning_team=ta, losing_team=tb)
            }
            for season, ta, tb in tqdm(iterable=tourney_games, desc='Recording predictions')
        ).set_index('ID').sort_index()

        print(f'Log Loss: {log_loss_error(predictions_df=predictions_df, access=access)}')

        predictions_df.to_csv(f'{access.prefix}SubmissionStage1.csv', index=True)


if __name__ == '__main__':
    main()
