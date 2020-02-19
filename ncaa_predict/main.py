from ncaa_predict.data.access import mens_access, womens_access
from ncaa_predict.features import registry
from ncaa_predict.data.processed import compact_team_results_df


def main():
    _compact_team_results_df = compact_team_results_df(womens_access)
    features_results = registry.run(access=womens_access)
    print(features_results)


if __name__ == '__main__':
    main()
