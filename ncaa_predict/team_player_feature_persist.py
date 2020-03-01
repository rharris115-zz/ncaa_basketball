from ncaa_predict.data.access import mens_access, womens_access
from ncaa_predict.features import tpf
import pandas as pd


def main():
    for access in (womens_access, mens_access):
        team_player_features_df = pd.concat(tpf.run(access=access).values(), axis=1)
        team_player_features_df.to_pickle(f'{access.prefix}TeamPlayerFeatures.pkl')


if __name__ == '__main__':
    main()
