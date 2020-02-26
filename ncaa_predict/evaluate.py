from ncaa_predict.data.access import DataAccess
from ncaa_predict.data.processed import ground_truth_since_2015
from sklearn.metrics import log_loss
import pandas as pd


def log_loss_error(predictions_df: pd.DataFrame, access: DataAccess) -> float:
    truth_df = ground_truth_since_2015(access)
    comparison_df = pd.concat([truth_df, predictions_df], axis=1, join='inner')
    comparison_df.to_csv(f'{access.prefix}ComparisonStage1.csv', index=True)
    loss = log_loss(y_true=comparison_df.Win, y_pred=comparison_df.Pred)
    return loss
