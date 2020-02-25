from ncaa_predict.data.access import mens_access, womens_access
from ncaa_predict.features import registry
from ncaa_predict.data.processed import all_team_results_df, possible_games
import pandas as pd


def main():
    d = registry.run(access=womens_access)
    team_features_df = pd.concat(d.values(), axis=1)
    print(team_features_df.Elo)


if __name__ == '__main__':
    main()
