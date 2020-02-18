import pandas as pd
import os, zipfile


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

    def cities_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name='Cities.csv')

    def conferences_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name='Conferences.csv')

    def conferences_tourney_games_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'ConferenceTourneyGames.csv')

    def game_cities_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'GameCities.csv')

    def tourney_compact_results_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'NCAATourneyCompactResults.csv')

    def tourney_detailed_results_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'NCAATourneyDetailedResults.csv')

    def tourney_seeds_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'NCAATourneySeeds.csv')

    def tourney_slots_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'NCAATourneySlots.csv')

    def regular_season_compact_results_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'RegularSeasonCompactResults.csv')

    def regular_season_detailed_results_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'RegularSeasonDetailedResults.csv')

    def seasons_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'Seasons.csv')

    def team_conferences_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'TeamConferences.csv')

    def teams_df(self) -> pd.DataFrame:
        return self._read_stage_1_file(name=self.prefix + 'Teams.csv')

    def events_df(self, season: int) -> pd.DataFrame:
        return self._read(name=f'{self.prefix}Events{season}.csv')

    def players_df(self) -> pd.DataFrame:
        return self._read(name=f'{self.prefix}Players.csv')


mens_access = DataAccess(zip_file='google-cloud-ncaa-march-madness-2020-division-1-mens-tournament.zip',
                         prefix='M')
womens_access = DataAccess(zip_file='google-cloud-ncaa-march-madness-2020-division-1-womens-tournament.zip',
                           prefix='W')
