''' 2024-02-08
Import this module and use its methods to deal with data from MDH.
Instance takes in dataframe and also useful for storing metadata and other forms of data.
All methods are used for computing data from 1 vessel only.
correct_time func is first used to compute the mean_idle time and for others.
The sequence is as follows, and correct_time func must be performed 1 more time after slice year func.
Adjust attribute self.long_trip_flag for long time spent at port threshold.
'''
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

# truth table. The list inside the keys are the flags for incomplete movement
truthTable = {'A':['A','T'], 
             'I':['A','T'], 
             'D':['I','D'], 
             'T':['I','D']}

class mdh_data_processor():
    
    def __init__(self,_df):
        self.raw_df = _df
        self.df = self.raw_df
        self.imo = self.df.loc[0,'imoNumber']
        self.name = self.df.loc[0,'vesselName']
        self.rows = len(self.df)
        self.long_trip_flag = 10
        self.dt_na_count = 0
        self.loc_na_count = 0
        self.negative_count = 0
        self.incomplete_count = 0
        self.slice_year_count = 0
        self.mean_idle_per_trip = 0
        self.total_hrs_by_year = 0
        self.moving_hrs_by_year = 0
        self.idling_hrs_by_year = 0
        self.hrs_per_trip_by_year = 0
        self.flag = 'OK'
        self.perform_all_corrections()
        
    def clean(self):
        # take useful cols
        self.df.rename(columns={'movementStartDateTime':'start','movementEndDateTime':'end',
                           'locationFrom':'from','locationTo':'to','movementType':'mtype'},inplace=True)
        self.df = self.df[['start','end','from','to','mtype',]]
        # get sum of all dt nulls
        self.dt_na_count = self.df.isna()[['start','end']].sum().sum()
        # get sum of all loc nulls
        self.loc_na_count = self.df.isna()[['from','to']].sum().sum()
        # handle all kinds of nulls for start and end
        self.df = self.df.drop( self.df [ (self.df['start'].isna()) & (self.df['end'].isna()) ].index )
        # assign null end dt with the start dt
        self.df.loc[self.df.isna()['end'],'end'] = self.df.loc[self.df.isna()['end'],'start']
        # assign null start dt with the end dt
        self.df.loc[self.df.isna()['start'],'start'] = self.df.loc[self.df.isna()['start'],'end']
        # as there were drops, index reset is required
        self.df.reset_index(drop=True,inplace=True)
        # convert all dt to dt obj
        self.df['start'] = pd.to_datetime(self.df['start'])
        self.df['end'] = pd.to_datetime(self.df['end'])
        # create new col for year identification
        self.df['year'] = self.df['start'].dt.year
        # drop all mtype of T
        self.df = self.df.drop(self.df[self.df['mtype']=='T'].index)
        self.df.reset_index(drop=True, inplace=True)
        
    def correct_time(self):
        ''' get time difference from movement '''
        self.df['timediff'] = self.df['end'] - self.df['start']
        ''' get idle
        pointer starts at 0, checks with row below.
        pointer stops at 2nd last row '''
        for index in range(len(self.df)-1):
            self.df.loc[index,'idle'] = self.df.loc[index+1,'start'] - self.df.loc[index,'end']
        # input zero data into last entry (nothing to subtract from)
        self.df.loc[len(self.df)-1,'idle'] = timedelta(days=0)
        '''corrections based on outside of sg, D and T '''
        for index, mtype in enumerate(self.df['mtype']):
            if mtype == 'D':
                # For D, if vessel previous row is T, timediff is not caounted
                self.df.loc[index,'idle'] = timedelta(days=0)
            elif mtype == 'T':
                self.df.loc[index,'idle'] = timedelta(days=0)
                self.df.loc[index,'timediff'] = timedelta(days=0)
        ''' corrections -ve timediff and idle '''
        # some timediff and idle are -ve (probably typo or what), set to zero
        self.negative_count = self.df[self.df['timediff']<timedelta(days=0)]['timediff'].count()
        self.df.loc[self.df['timediff'] < timedelta(days=0),'timediff'] = timedelta(days=0)
        self.df.loc[self.df['idle'] < timedelta(days=0),'idle'] = timedelta(days=0)

    def insert_incomplete_journeys(self):
        ''' Insert rows with null values to complete journey
        pointer starts at 1, checks with row above.
        pointer stops on last row.
        collect all affected pair indexes into a list. 
        '''
        for index, mtype in enumerate(self.df['mtype'].iloc[1:len(self.df)],start=1):
            if mtype in truthTable[self.df.loc[index-1,'mtype']]:
                if self.df.iloc[index-1]['mtype'] == 'A' or self.df.iloc[index-1]['mtype'] == 'I':
                    # NOTE: this order & the next is important to how you prepare the data
                    self.df.loc[index-0.5,self.df.columns] = [np.nan, np.nan, 'INSIDE','OUTSIDE', 
                                                    'D', np.nan, np.nan, np.nan]
                    self.incomplete_count += 1
                elif self.df.iloc[index-1]['mtype'] == 'D' or self.df.iloc[index-1]['mtype'] == 'T':
                    self.df.loc[index-0.5,self.df.columns] = [np.nan, np.nan, 'OUTSIDE','INSIDE',
                                                    'A', np.nan, np.nan, np.nan]
                    self.incomplete_count += 1
        # sort and reset index to insert those new rows into the affected row pairs
        self.df = self.df.sort_index().reset_index(drop=True)

    def sort_trips(self):
        '''creates a new col with trip numbers
        if mtype is A, adds 1 to trip '''
        trips = [0,]
        for mtype in self.df['mtype']:
            if mtype == 'A':
                trips[-1] = trips[-1]+1
            trips.append(trips[-1])
        trips.pop(-1)
        self.df['trip'] = np.array(trips)

    def get_mean_idle_per_trip(self):
        ''' looks at legit trips (those w/o nan)
        computes their mean idle per trip '''
        trips_nan = self.df[self.df['idle'].isna()]['trip']
        trips_all = self.df['trip'].unique()
        trips_legit = [trip for trip in trips_all if trip not in trips_nan.values]
        # filter by legit trips
        df_legit = self.df[self.df['trip'].isin(trips_legit)]
        df_legit.groupby(by='trip')['idle'].sum()
        self.mean_idle_per_trip = df_legit.groupby(by='trip')['idle'].sum().mean()
        # round off to nearnest min
        self.mean_idle_per_trip = self.mean_idle_per_trip.round('min')
        # raise flag when hours gets too long
        if self.mean_idle_per_trip > timedelta(days = self.long_trip_flag):
            self.flag = 'LONG'

    def correct_incomplete_journeys(self):
        ''' For all null values
        for mtype 'D':  add mean idle time to last end, to now start
                        now end to be same as now start
        for mtype 'A':  now end = later start - mean_idle, now start = now end
                        timediff, idle = zero
        '''
        for index in self.df[self.df['start'].isna()].index:
            if self.df.loc[index,'mtype'] == 'D':
                self.df.loc[index,'start'] = self.df.loc[index-1,'end'] + self.mean_idle_per_trip
                self.df.loc[index,'end'] = self.df.loc[index,'start']
                self.df.loc[index,'year'] = self.df.loc[index,'end'].year
                self.df.loc[index,'timediff'] = timedelta(days=0)
                self.df.loc[index,'idle'] = timedelta(days=0)
                self.df.loc[index-1,'idle'] = self.mean_idle_per_trip
            if self.df.loc[index,'mtype'] == 'A':
                self.df.loc[index,'end'] = self.df.loc[index+1,'start'] - self.mean_idle_per_trip
                self.df.loc[index,'start'] = self.df.loc[index,'end'] 
                self.df.loc[index,'year'] = self.df.loc[index,'end'].year
                self.df.loc[index,'timediff'] = timedelta(days=0)
                self.df.loc[index,'idle'] = self.mean_idle_per_trip
        #return self.df

    def correct_last_incomplete_journey(self):
        ''' check last row mtype for termination '''
        last = len(self.df)-1
        if self.df.loc[last,'mtype'] == 'A' or self.df.loc[last,'mtype'] == 'I':
            self.df.loc[len(self.df),self.df.columns] = [self.df.loc[last,'start'], self.df.loc[last,'end'],'INSIDE','OUTSIDE',
                                                         'D', self.df.loc[last,'year'], timedelta(days=0), timedelta(days=0), self.df.loc[last,'trip']]
            self.incomplete_count += 1

    def slice_year(self):
        ''' when crossing to next year and prior mtype is still A or I
        inserts a new row to slice year while preserving data for annual calculations
        need to perform correct_time func one more time
        pointer starts at 1, checks with row above.
        pointer stops on last row.
        '''
        for index, year in enumerate(self.df['year'].iloc[1:len(self.df)],start=1):
            if year != self.df.loc[index-1,'year']:
                if self.df.loc[index-1,'mtype'] == 'A' or self.df.loc[index-1,'mtype'] == 'I':
                    thisYearStart = datetime(int(year), 1, 1, 0, 0, 0)
                    # input for last year end
                    self.df.loc[index-0.5, self.df.columns] = [thisYearStart, thisYearStart, 'INTERIM','INTERIM', 
                                                     'I', self.df.loc[index,'year'], timedelta(days=0), np.nan, self.df.loc[index,'trip'] ]
                    self.slice_year_count += 1
        # sort and reset index to insert those new rows into the affected row pairs 
        self.df = self.df.sort_index().reset_index(drop=True)
        self.df['trip'] = self.df['trip'].astype(int)
        #return self.df
    
    def perform_all_corrections(self):
        self.clean()
        self.correct_time()
        self.insert_incomplete_journeys()
        self.sort_trips()
        self.get_mean_idle_per_trip()
        self.correct_incomplete_journeys()
        self.correct_last_incomplete_journey()
        self.slice_year()
        self.correct_time()
    
    def get_moving_hours_by_year(self,year):
        self.moving_hrs_by_year = self.df[self.df['year']==year]['timediff'].sum()
        self.moving_hrs_by_year = self.moving_hrs_by_year.total_seconds() / 3600
        self.moving_hrs_by_year = round(self.moving_hrs_by_year,2)
        return self.moving_hrs_by_year
        
    def get_idling_hours_by_year(self,year):
        self.idling_hrs_by_year = self.df[self.df['year']==year]['idle'].sum()
        self.idling_hrs_by_year = self.idling_hrs_by_year.total_seconds() / 3600
        self.idling_hrs_by_year = round(self.idling_hrs_by_year,2)
        return self.idling_hrs_by_year
        
    def get_total_hours_by_year(self,year):
        self.moving_hrs_by_year = self.get_moving_hours_by_year(year)
        self.idling_hrs_by_year = self.get_idling_hours_by_year(year)
        self.total_hrs_by_year = round(self.moving_hrs_by_year + self.idling_hrs_by_year, 2)
        return self.total_hrs_by_year

    def get_all_hours_by_year(self,year):
        self.total_hrs_by_year = self.get_total_hours_by_year(year)
        return self.moving_hrs_by_year, self.idling_hrs_by_year, self.total_hrs_by_year

    def get_no_of_trips_by_year(self,year):
        df_by_year = self.df[self.df['year'] == year]
        no_of_trips_by_year = len(df_by_year['trip'].unique())
        return no_of_trips_by_year
        
    def get_hours_per_trip_by_year(self,year):
        no_of_trips = self.get_no_of_trips_by_year(year)
        self.hrs_per_trip_by_year = round( self.total_hrs_by_year / no_of_trips , 2)
        return self.hrs_per_trip_by_year

    def show_report_by_year(self,year):
        ''' Show annual report in a nice table form. '''
        df_by_year = self.df[self.df['year'] == year]
        # create show df
        df_show = pd.DataFrame(columns=['Trip','Arrived','Departed','Moving','Idle','Moving+Idle'])
        df_show['Trip'] = df_by_year['trip'].unique()
        df_show['Arrived'] = df_by_year.groupby(by='trip')['start'].min().values
        df_show['Departed'] = df_by_year.groupby(by='trip')['end'].max().values
        df_show['Moving'] = df_by_year.groupby(by='trip')['timediff'].sum().values
        df_show['Idle'] = df_by_year.groupby(by='trip')['idle'].sum().values
        durations = df_by_year.groupby(by='trip')['timediff'].sum() + df_by_year.groupby(by='trip')['idle'].sum()
        df_show['Moving+Idle'] = durations.values
        last_entry_complete = df_by_year.iloc[-1]['mtype'] == 'D' or df_by_year.iloc[-1]['mtype'] == 'T'
        total_trips = len(df_show)
        # derive
        total_moving, mean_moving = df_show['Moving'].sum(), df_show['Moving'].mean().round('min')
        total_idle, mean_idle = df_show['Idle'].sum(), df_show['Idle'].mean().round('min')
        df_show.loc[len(df_show)] = 'Total Duration', '-', '-', total_moving, total_idle , (total_moving + total_idle)
        df_show.loc[len(df_show)] = 'Mean Duration', '-', '-', mean_moving, mean_idle , (mean_moving + mean_idle)
        df_show.loc[len(df_show)] = 'Total Trips', '-', '-', '-', '-', total_trips
        df_show.loc[len(df_show)] = 'Last trip complete?', '-', '-', '-', '-', last_entry_complete
        # create fig
        fig = plt.figure(figsize = (12, .5))
        ax = fig.add_subplot(111)
        #create table
        ax.table(cellText = df_show.values, colLabels = df_show.columns, cellLoc='center')
        ax.set_title('Trip report breakdown\nImo: '+ str(self.imo) + '\nName: ' + self.name + '\nYear ' + str(int(year)))
        ax.axis('off')

    def show_error_report(self):
        ''' Show error report in a nice table form, across whole dataset. '''
        data={'start end nulls':self.dt_na_count,
              'location nulls':self.loc_na_count,
              'negative travel time':self.negative_count,
              'incomplete trips':self.incomplete_count,
              'trip cross over year': self.slice_year_count,
              'mean idling per trip':self.mean_idle_per_trip,
              'idle < {}?'.format(self.long_trip_flag): self.flag}
        self.df_error = pd.DataFrame.from_dict(data, orient='index', columns=['count'])
        fig = plt.figure(figsize = (2,.6))
        ax = fig.add_subplot(111)
        #create table
        ax.table(cellText = self.df_error.values, rowLabels = self.df_error.index, colLabels = self.df_error.columns, cellLoc='center')
        ax.set_title('Error Log\nImo: '+ str(self.imo) + '\nName: ' + self.name)
        ax.axis('off')