from . import tpf
from ..data.access import DataAccess, player_features_df
import pandas as pd
from ..data.processed import to_team_format, game_format_indices
import numpy as np
from tqdm import tqdm


@tpf.register
def scoring_entropy(access: DataAccess) -> pd.DataFrame:
    pf_df = player_features_df(prefix=access.prefix).reset_index()
    pf_df['Score'] = pf_df.made3 * 3 + pf_df.made2 * 2 + pf_df.made1

    def _entropy(df: pd.DataFrame) -> float:
        total_score = df.Score.sum()
        score_fraction = df.Score[df.Score > 0] / total_score
        return -(score_fraction * np.log(score_fraction)).sum()

    def _t(df: pd.DataFrame):
        winning_df = df[df.EventTeamID == df.WTeamID]
        losing_df = df[df.EventTeamID == df.LTeamID]
        return pd.Series(dict(WScoringEntropy=_entropy(winning_df), LScoringEntropy=_entropy(losing_df)))

    tqdm.pandas(desc="Calculating Scoring Entropy")
    scoring_entropy_df = pf_df.groupby(game_format_indices).progress_apply(_t)
    return to_team_format(game_formatted_df=scoring_entropy_df)
