import pandas as pd
import os, zipfile
from ..utils import memoize


class DataAccess():

    def __init__(self, zip_file: str, prefix: str):
        self.zip_file = zip_file
        self.prefix = prefix

    def _zip(self):
        return zipfile.ZipFile(os.path.join('.', self.zip_file))

    def _read(self, name: str) -> pd.DataFrame:
        zf = self._zip()
        _df = pd.read_csv(zf.open(name))
        return _df

    def _read_stage_1_file(self, name: str) -> pd.DataFrame:
        zf = self._zip()
        _df = pd.read_csv(zf.open(os.path.join(self.prefix + 'DataFiles_Stage1', name)))
        return _df

    @memoize
    def cities_df(self) -> pd.DataFrame:
        # CityID, City, State
        return self._read_stage_1_file(name='Cities.csv')

    @memoize
    def conferences_df(self) -> pd.DataFrame:
        # ConfAbbrev, Description
        return self._read_stage_1_file(name='Conferences.csv')

    @memoize
    def conferences_tourney_games_df(self) -> pd.DataFrame:
        # Season, ConfAbbrev, DayNum, WTeamID, LTeamID
        return self._read_stage_1_file(name=self.prefix + 'ConferenceTourneyGames.csv')

    @memoize
    def game_cities_df(self) -> pd.DataFrame:
        # Season, DayNum, WTeamID, LTeamID, CRType, CityID
        return self._read_stage_1_file(name=self.prefix + 'GameCities.csv')

    @memoize
    def tourney_compact_results_df(self) -> pd.DataFrame:
        # Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc,NumOT
        return self._read_stage_1_file(name=self.prefix + 'NCAATourneyCompactResults.csv')

    @memoize
    def tourney_detailed_results_df(self) -> pd.DataFrame:
        # Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT, WFGM, WFGA, WFGM3, WFGA3, WFTM, WFTA, WOR, WDR, WAst, WTO, WStl, WBlk, WPF, LFGM, LFGA, LFGM3, LFGA3, LFTM, LFTA, LOR, LDR, LAst, LTO, LStl, LBlk, LPF
        return self._read_stage_1_file(name=self.prefix + 'NCAATourneyDetailedResults.csv')

    @memoize
    def tourney_seeds_df(self) -> pd.DataFrame:
        # Season, Seed, TeamID
        return self._read_stage_1_file(name=self.prefix + 'NCAATourneySeeds.csv')

    @memoize
    def tourney_slots_df(self) -> pd.DataFrame:
        # Slot, StrongSeed, WeakSeed
        return self._read_stage_1_file(name=self.prefix + 'NCAATourneySlots.csv')

    @memoize
    def regular_season_compact_results_df(self) -> pd.DataFrame:
        # Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT
        return self._read_stage_1_file(name=self.prefix + 'RegularSeasonCompactResults.csv')

    @memoize
    def regular_season_detailed_results_df(self) -> pd.DataFrame:
        # Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT, WFGM, WFGA, WFGM3, WFGA3, WFTM, WFTA, WOR, WDR, WAst, WTO, WStl, WBlk, WPF, LFGM, LFGA, LFGM3, LFGA3, LFTM, LFTA, LOR, LDR, LAst, LTO, LStl, LBlk, LPF
        return self._read_stage_1_file(name=self.prefix + 'RegularSeasonDetailedResults.csv')

    @memoize
    def seasons_df(self) -> pd.DataFrame:
        # Season, DayZero, RegionW, RegionX, RegionY, RegionZ
        return self._read_stage_1_file(name=self.prefix + 'Seasons.csv')

    @memoize
    def team_conferences_df(self) -> pd.DataFrame:
        # Season, TeamID, ConfAbbrev
        return self._read_stage_1_file(name=self.prefix + 'TeamConferences.csv')

    @memoize
    def teams_df(self) -> pd.DataFrame:
        # TeamID, TeamName
        return self._read_stage_1_file(name=self.prefix + 'Teams.csv')

    @memoize
    def events_df(self, season: int) -> pd.DataFrame:
        # EventID, Season, DayNum, WTeamID, LTeamID, WFinalScore, LFinalScore, WCurrentScore, LCurrentScore, ElapsedSeconds, EventTeamID, EventPlayerID, EventType, EventSubType, X, Y, Area
        return self._read(name=f'{self.prefix}Events{season}.csv')

    @memoize
    def players_df(self) -> pd.DataFrame:
        # PlayerID, LastName, FirstName, TeamID
        return self._read(name=f'{self.prefix}Players.csv')

    @memoize
    def sample_submission_stage_1(self):
        # ID, Pred
        return self._read(name=f'{self.prefix}SampleSubmissionStage1_2020.csv')


mens_access = DataAccess(zip_file='google-cloud-ncaa-march-madness-2020-division-1-mens-tournament.zip',
                         prefix='M')
womens_access = DataAccess(zip_file='google-cloud-ncaa-march-madness-2020-division-1-womens-tournament.zip',
                           prefix='W')


@memoize
def team_features_df(prefix: str) -> pd.DataFrame:
    return pd.read_pickle(f'{prefix}TeamFeatures.pkl')


@memoize
def player_features_df(prefix: str) -> pd.DataFrame:
    return pd.read_pickle(f'{prefix}PlayerFeatures.pkl')
