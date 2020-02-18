from ncaa_predict.data import access


def main():
    df = access.mens_access.conferences_tourney_games_df()
    print(df)


if __name__ == '__main__':
    main()
