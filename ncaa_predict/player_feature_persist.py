from ncaa_predict.data.access import mens_access, womens_access
from ncaa_predict.features import pf
import pandas as pd


def main():
    for access in (mens_access, womens_access):
        player_features_df = pd.concat(pf.run(access=access).values(), axis=1)
        player_features_df.to_pickle(f'{access.prefix}PlayerFeatures.pkl')


if __name__ == '__main__':
    main()
