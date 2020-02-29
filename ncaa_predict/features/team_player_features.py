from . import tpf
from ..data.access import DataAccess, player_features_df
import pandas as pd
from ..data.processed import player_game_format_indices


@tpf.register
def scoring_entropy(access: DataAccess) -> pd.Series:
    pf_df = player_features_df(prefix=access.prefix)
    pf_df['Score'] = pf_df.made3 * 3 + pf_df.made2 * 2 + pf_df.made1
    pass
