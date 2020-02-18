from ncaa_predict.data.access import mens_access, womens_access


def main():
    df = mens_access.players_df()
    print(df)


if __name__ == '__main__':
    main()
