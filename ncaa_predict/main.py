from ncaa_predict.data.access import mens_access
from ncaa_predict.data.processed import regular_season_compact_team_results_df


def main():
    df = regular_season_compact_team_results_df(access=mens_access)
    print(df)


if __name__ == '__main__':
    main()
