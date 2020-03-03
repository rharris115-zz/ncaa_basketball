from . import tpf
from ..data.access import DataAccess, player_features_df
import pandas as pd
from ..data.processed import to_team_format, game_format_indices
import numpy as np


@tpf.register
def assist_entropy(access: DataAccess) -> pd.Series:
    pf_df = player_features_df(prefix=access.prefix).reset_index()

    w_total_assists = pf_df[pf_df.EventTeamID == pf_df.WTeamID].groupby(game_format_indices) \
        .assist.sum().rename('WTotalAssist')
    l_total_assists = pf_df[pf_df.EventTeamID == pf_df.LTeamID].groupby(game_format_indices) \
        .assist.sum().rename('LTotalAssist')

    pf_df = pf_df.merge(w_total_assists, on=game_format_indices, how='outer')
    pf_df = pf_df.merge(l_total_assists, on=game_format_indices, how='outer')

    pf_df['WAssistFraction'] = np.where(pf_df.EventTeamID == pf_df.WTeamID, pf_df.assist / pf_df.WTotalAssist, 0)
    pf_df['LAssistFraction'] = np.where(pf_df.EventTeamID == pf_df.LTeamID, pf_df.assist / pf_df.LTotalAssist, 0)

    pf_df['WAssistEntropyContribution'] = -pf_df.WAssistFraction * np.log(pf_df.WAssistFraction)
    pf_df['LAssistEntropyContribution'] = -pf_df.LAssistFraction * np.log(pf_df.LAssistFraction)

    w_assist_entropy = pf_df.groupby(game_format_indices).WAssistEntropyContribution.sum().rename('WAssistEntropy')
    l_assist_entropy = pf_df.groupby(game_format_indices).LAssistEntropyContribution.sum().rename('LAssistEntropy')

    assist_entropy_game_format_df = pd.concat([w_assist_entropy, l_assist_entropy], axis=1)
    assist_entropy_df = to_team_format(game_formatted_df=assist_entropy_game_format_df)
    return assist_entropy_df.AssistEntropy


@tpf.register
def scoring_entropy(access: DataAccess) -> pd.Series:
    pf_df = player_features_df(prefix=access.prefix).reset_index()
    pf_df['Score'] = pf_df.made3 * 3 + pf_df.made2 * 2 + pf_df.made1

    pf_df['WScoreFraction'] = np.where(pf_df.EventTeamID == pf_df.WTeamID, pf_df.Score / pf_df.WFinalScore, 0)
    pf_df['LScoreFraction'] = np.where(pf_df.EventTeamID == pf_df.LTeamID, pf_df.Score / pf_df.LFinalScore, 0)

    pf_df['WScoreEntropyContribution'] = -pf_df.WScoreFraction * np.log(pf_df.WScoreFraction)
    pf_df['LScoreEntropyContribution'] = -pf_df.LScoreFraction * np.log(pf_df.LScoreFraction)

    w_scoring_entropy = pf_df.groupby(game_format_indices).WScoreEntropyContribution.sum().rename('WScoringEntropy')
    l_scoring_entropy = pf_df.groupby(game_format_indices).LScoreEntropyContribution.sum().rename('LScoringEntropy')

    scoring_entropy_game_format_df = pd.concat([w_scoring_entropy, l_scoring_entropy], axis=1)
    scoring_entropy_df = to_team_format(game_formatted_df=scoring_entropy_game_format_df)
    return scoring_entropy_df.ScoringEntropy
