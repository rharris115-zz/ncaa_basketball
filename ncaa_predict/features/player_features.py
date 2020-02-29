import pandas as pd

from . import pf
from ..data.access import DataAccess
from ..data.processed import player_stats_df

# assist - an assist was credited on a made shot
# block - a blocked shot was recorded
# steal - a steal was recorded
# sub - a substitution occurred, with one of the following subtypes: in=player entered the game; out=player exited the game; start=player started the game
# timeout - a timeout was called, with one of the following subtypes: unk=unknown type of timeout; comm=commercial timeout; full=full timeout; short= short timeout
# turnover - a turnover was recorded, with one of the following subtypes: unk=unknown type of turnover; 10sec=10 second violation; 3sec=3 second violation; 5sec=5 second violation; bpass=bad pass turnover; dribb=dribbling turnover; lanev=lane violation; lostb=lost ball; offen=offensive turnover (?); offgt=offensive goaltending; other=other type of turnover; shotc=shot clock violation; trav=travelling
# foul - a foul was committed, with one of the following subtypes: unk=unknown type of foul; admT=administrative technical; benT=bench technical; coaT=coach technical; off=offensive foul; pers=personal foul; tech=technical foul
# fouled - a player was fouled
# reb - a rebound was recorded, with one of the following subtypes: deadb=a deadball rebound; def=a defensive rebound; defdb=a defensive deadball rebound; off=an offensive rebound; offdb=an offensive deadball rebound
# made1, miss1 - a one-point free throw was made or missed, with one of the following subtypes: 1of1=the only free throw of the trip to the line; 1of2=the first of two free throw attempts; 2of2=the second of two free throw attempts; 1of3=the first of three free throw attempts; 2of3=the second of three free throw attempts; 3of3=the third of three free throw attempts; unk=unknown what the free throw sequence is
# made2, miss2 - a two-point field goal was made or missed, with one of the following subtypes: unk=unknown type of two-point shot; dunk=dunk; lay=layup; tip=tip-in; jump=jump shot; alley=alley-oop; drive=driving layup; hook=hook shot; stepb=step-back jump shot; pullu=pull-up jump shot; turna=turn-around jump shot; wrong=wrong basket
# made3, miss3 - a three-point field goal was made or missed, with one of the following subtypes: unk=unknown type of three-point shot; jump=jump shot; stepb=step-back jump shot; pullu=pull-up jump shot; turna=turn-around jump shot; wrong=wrong basket
# jumpb - a jumpball was called or resolved, with one of the following subtypes: start=start period; block=block tie-up; heldb=held ball; lodge=lodged ball; lost=jump ball lost; outof=out of bounds; outrb=out of bounds rebound; won=jump ball won

scoring_event_types = ['made1', 'made2', 'made3', 'miss1', 'miss2', 'miss3']
offensive_event_types = ['assist', 'turnover', 'fouled']
defensive_event_types = ['block', 'steal', 'reb', 'foul']


@pf.register
def player_scoring_stats_df(access: DataAccess) -> pd.DataFrame:
    return player_stats_df(*scoring_event_types, access=access)


@pf.register
def player_offensive_stats_df(access: DataAccess) -> pd.DataFrame:
    return player_stats_df(*offensive_event_types, access=access)


@pf.register
def player_defensive_stats_df(access: DataAccess) -> pd.DataFrame:
    return player_stats_df(*defensive_event_types, access=access)
