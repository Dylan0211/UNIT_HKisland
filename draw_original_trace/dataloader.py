from config import *
from datetime import datetime
from make_cl_sum import *

import pickle
import pandas as pd


def normalize(df):
    load_min = df['coolingLoad'].min()
    load_max = df['coolingLoad'].max()
    df['nor_cl'] = 0
    df['nor_cl'] = (df['coolingLoad'] - load_min) / (load_max - load_min)

    df['nor_temp'] = 0
    df['nor_temp'] = (df['temperature'] - temperature_min) / (temperature_max - temperature_min)

    return df, load_min, load_max


def build_original_hk_island_load_table(df):
    begin_time = df.loc[0, 'time']
    end_time = df.loc[df.shape[0] - 1, 'time']
    should_total_hours = (datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S") -
                          datetime.strptime(begin_time, "%Y-%m-%d %H:%M:%S")).days * 24

    time_col = pd.date_range(begin_time, periods=should_total_hours, freq='1h')
    df2 = pd.DataFrame(columns=['time'])
    df2['time'] = time_col
    df['time'] = df['time'].astype('datetime64')
    df3 = pd.merge(df2, df, how='outer', on=['time'])

    df3.sort_values('time', inplace=True)
    df3.reset_index(drop=True, inplace=True)
    df3.fillna(0, inplace=True)
    return df3[['time', 'coolingLoad', 'temperature']]


def fill_value_into_null(df):
    for index in range(df.shape[0]):
        if df.loc[index, 'temperature'] == 0:
            try:
                df.loc[index, 'temperature'] = df.loc[index - 1, 'temperature']
            except:
                pass

        if df.loc[index, 'coolingLoad'] == 0:
            try:
                df.loc[index, 'coolingLoad'] = df.loc[index - 24, 'coolingLoad']
            except:
                pass
    return df


def read_data(building_name):
    # load data
    csv_path = '../raw_data/'
    building_csv_path = csv_path + building_name + '.csv'
    print('csv path: {}'.format(building_csv_path))
    df = make_big_df(building_csv_path)

    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # fill in missing val
    print(df.shape)
    df = build_original_hk_island_load_table(df=df)
    df = fill_value_into_null(df=df)
    print(df.shape)

    # normalize
    df, load_min, load_max = normalize(df=df)

    # save
    print('data shape: {}'.format(df.shape))
    with open('./tmp_pkl_data/{}_max_min_context.pkl'.format(building_name), 'wb') as w:
        pickle.dump((load_max, load_min, df), w)


if __name__ == '__main__':
    read_data(building_name=building_name)
