import pandas as pd

from . import pf
from ..data.access import DataAccess
from ..data.processed import extract_player_playing_time


@pf.register
def playing_time(access: DataAccess) -> pd.DataFrame:
    events_df = access.events_df(season=2015)
    extract_player_playing_time(events_df=events_df)
