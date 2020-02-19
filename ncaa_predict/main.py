from ncaa_predict.data.access import mens_access
from ncaa_predict.data.processed import regular_season_compact_team_results_df
from ncaa_predict.features import mens_features, womens_features


def main():
    features_results = mens_features.run()
    print(features_results)


if __name__ == '__main__':
    main()
