import os

import pandas as pd

from ncaa_predict.data.access import mens_access, womens_access, team_features_df, team_player_features
from ncaa_predict.data.processed import possible_games, infer_slot_dates
from ncaa_predict.evaluate import log_loss_error
from ncaa_predict.models.prediction import tournament_game_index_labels, EloTournamentPredictor, LRTournamentPredictor, \
    MLPTournamentPredictor


def main():
    for access in (mens_access, womens_access):
        tpf_df = team_player_features(prefix=access.prefix)
        tf_df = team_features_df(prefix=access.prefix)

        tf_df = tf_df.join(tpf_df, how='inner')

        tourney_games_df = pd.DataFrame.from_records(
            {'Season': season, 'TeamID': team_id, 'OtherTeamID': other_team_id}
            for season, team_id, other_team_id in possible_games(access)
            if season >= 2015
        )

        slot_dates = infer_slot_dates(access=access)

        for pred in (
                EloTournamentPredictor(),
                LRTournamentPredictor(),
                MLPTournamentPredictor(),
        ):
            pred.train(team_features_df=tf_df)
            pred_name = type(pred).__name__

            predictions_df = pred.estimate_probability(tourney_games_df=tourney_games_df).reset_index()

            predictions_df['ID'] = predictions_df.Season.astype(str) + '_' \
                                   + predictions_df.TeamID.astype(str) + '_' \
                                   + predictions_df.OtherTeamID.astype(str)
            predictions_df.drop(columns=tournament_game_index_labels, inplace=True)
            predictions_df.set_index('ID', inplace=True)
            predictions_df.sort_index(inplace=True)

            log_loss = log_loss_error(predictions_df=predictions_df,
                                      access=access,
                                      comparison_file=os.path.join('out',
                                                                   f'{access.prefix}_{pred_name}_ComparisonStage1.csv'))
            print(f'{access.prefix} {pred_name} Log Loss: {log_loss}')

            predictions_df.to_csv(os.path.join('out', f'{access.prefix}_{pred_name}_SubmissionStage1.csv'),
                                  index=True)


if __name__ == '__main__':
    main()
