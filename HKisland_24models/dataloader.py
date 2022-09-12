"""
不要直接跑这个dataloader，跑每个model里面的read_data()
"""
from datetime import datetime

import pickle
import numpy as np
import pandas as pd
import time


def one_hot(data, feature):
    onehot = pd.get_dummies(data[feature])
    df = data.drop(feature, axis=1)
    data = df.join(onehot)
    return data


def normalize(df, temperature_min, temperature_max):
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


def make_big_df(path):
    # 1. load csv data
    # data = pd.read_csv('data_27012021/spot_objs20210127.csv', header=None, index_col=0)
    data = pd.read_csv(path)  # 这份data最多同时刻开三台
    print(data.shape)
    print('chillerName:', data['chillerName'].unique())
    data = data.loc[data['cop'] > 0, :].reset_index(drop=True)
    print('过滤cop=0的data, shape:', data.shape)

    data.reset_index(drop=True, inplace=True)
    print(data.shape)
    count = data['time'].value_counts()
    print('max run simultaneously:', count[0])

    # data['time_stamp'] = pd.to_datetime(data['time'], unit='s')

    # 2. get each chiller data, and adjust columns naming
    chillers_list = []
    for i in data['chillerName'].unique():
        data_i = data[data['chillerName'] == i]
        if data_i.shape[0] <= 0:
            break
        print('i=', i, 'data_i.shape = ', data_i.shape)
        columns = data_i.columns.to_list()
        for j in range(2, len(columns)):
            columns[j] = columns[j] + '_' + str(i)
        data_i.columns = columns
        chillers_list.append(data_i)

    # 3. 把上面的的子表全部拼起来
    df_big = chillers_list[0]
    for i in range(len(chillers_list) - 1):
        df_big = pd.merge(df_big, chillers_list[i + 1], on=['time', 'building'], how='outer')

    df_big.sort_values('time', inplace=True)
    df_big.reset_index(drop=True, inplace=True)
    print('after merge by time:', df_big.shape)

    # 对每行求load总和
    df_big['coolingLoad'] = 0
    load_columns_name_list = [item for item in df_big.columns.to_list() if 'coolingLoad' in item]
    df_loads = df_big[load_columns_name_list]
    df_loads = df_loads.fillna(0)
    df_big['coolingLoad'] = df_loads.sum(axis=1)

    df_big['temperature'] = 0
    temp_columns_name_list = [item for item in df_big.columns.to_list() if 'temperature' in item]
    df_temp = df_big[temp_columns_name_list]
    df_temp = df_temp.fillna(0)
    df_big['temperature'] = df_temp.max(axis=1)

    return df_big[['time', 'coolingLoad', 'temperature']]


def context_setting_weather_day(df):
    """
    被pandas map调用
    类似复现 IJCAI：df["hour_level"] = df["hour"].map(time_hour_map)
    x 就直接就是time，不是一整行df

    # 这里用的context是 Ashrae 的Weather day type 24-hour profile plots 提到的8分类：

    winter peak weekday,                0
    winter average weekday,             1
    winter average weekend day/holiday, 2

    summer peak weekday,                3
    summer average weekday,             4
    summer average weekend day/holiday, 5

    spring average weekday,             6  todo 这里有个问题，没有spring的weekend
    fall average weekday                7

    :param x:
    :return:
    """

    #  0. 新添加的几列
    df['8_class'] = -1  # 最后要给每条item填上对应的class
    df['season'] = -1
    df['weekend_flag'] = -1

    # 1. season
    seasons = {
        1: 'Winter',
        2: 'Spring',
        3: 'Summer',
        4: 'Autumn'
    }

    def judge_season(x):
        return (x.month % 12 + 3) // 3

    df['season'] = df['time'].map(judge_season)

    # 2. is weekend？
    def judge_is_weekend_or_weekday(x):
        the_day = x.weekday()
        if the_day > 4:
            return 1  # weekend 是1
        else:
            return 0

    # here to add whether weekend
    df['weekend_flag'] = df['time'].map(judge_is_weekend_or_weekday)  # 0~6 从星期一开始到周日，0是星期一

    # 3. peak
    """
    For example, the summer peak weekday
    can be defined by selecting the five warmest non-holiday
    weekdays during June, July, and August using the actual
    weather data for the calibration period.
    """

    def get_year_column(x):
        return x.year

    top_peak_num = 20  # ashrae 里面的写的， todo 这个按每年的来
    df_temperature = df[['time', 'coolingLoad', 'temperature', 'season', 'weekend_flag']].copy()
    df_temperature['year'] = df_temperature['time'].map(get_year_column)
    for index in range(df_temperature.shape[0]):
        if df_temperature.loc[index, 'temperature'] == 0:  # load 有可能是真关机，但是天气不会==0
            df_temperature.loc[index, 'temperature'] = df_temperature.loc[index - 1, 'temperature']
    df_temperature['hour'] = df_temperature['time'].dt.hour
    df_temperature_12clock = df_temperature[df_temperature['hour'] == 12]  # 虽然12点不一定是一天中温度最高的时候，但是某一天12点的温度比其他天高，整体应该也比其他高

    # 3.1 winter peak weekday，这个应该是每年都有, todo 其实每年也有点不合理，但总比全部几年挑几天peak 合理
    the_peak_date_winter_list = []
    for year in pd.unique(df_temperature['year']):
        df_temperature_12clock_this_year = df_temperature_12clock[df_temperature_12clock['year'] == year]
        df_winter = df_temperature_12clock_this_year[df_temperature_12clock_this_year['season'] == 1].reset_index(drop=True)
        top_k_idx = np.array(df_winter['temperature']).argsort()[::-1][0: top_peak_num]
        the_peak_date_winter = df_winter.loc[top_k_idx, 'time']  # 装了气温最高的几天
        for ii in the_peak_date_winter.index:
            the_peak_date_winter_list.append(str(the_peak_date_winter.loc[ii]).split(' ')[0])  # item :'2018-08-01'

    # extract_wanted_days_data(days_list=the_peak_date_winter_list, df=df_temperature)

    # 3.2 summer peak weekday
    the_peak_date_summer_list = []
    for year in pd.unique(df_temperature['year']):
        df_temperature_12clock_this_year = df_temperature_12clock[df_temperature_12clock['year'] == year]
        df_summer = df_temperature_12clock_this_year[df_temperature_12clock_this_year['season'] == 3].reset_index(drop=True)
        top_k_idx = np.array(df_summer['temperature']).argsort()[::-1][0: top_peak_num]
        the_peak_date_summer = df_summer.loc[top_k_idx, 'time']  # 装了气温最高的几天
        for ii in the_peak_date_summer.index:
            the_peak_date_summer_list.append(str(the_peak_date_summer.loc[ii]).split(' ')[0])  # item :'2018-08-01'

    # 4. for 循环赋值
    print('build contextual column for df...')
    for index in range(df.shape[0]):
        # winter相关
        if df.loc[index, 'season'] == 1:
            if df.loc[index, 'weekend_flag'] == 1:
                df.loc[index, '8_class'] = 2
            if df.loc[index, 'weekend_flag'] == 0:
                df.loc[index, '8_class'] = 1

                # 这个也是要weekday
                if str(df.loc[index, 'time']).split(' ')[0] in the_peak_date_winter_list:
                    df.loc[index, '8_class'] = 0  # 这个放在前两个if后面，因为会有overwrite

        # summer 相关
        elif df.loc[index, 'season'] == 3:
            if df.loc[index, 'weekend_flag'] == 1:
                df.loc[index, '8_class'] = 5
            if df.loc[index, 'weekend_flag'] == 0:
                df.loc[index, '8_class'] = 4

                # 这个也是要weekday
                if str(df.loc[index, 'time']).split(' ')[0] in the_peak_date_summer_list:
                    df.loc[index, '8_class'] = 3  # 这个放在前两个if后面，因为会有overwrite

        # 春秋
        elif df.loc[index, 'season'] == 2:  # 春
            df.loc[index, '8_class'] = 6
        elif df.loc[index, 'season'] == 4:  # 秋
            df.loc[index, '8_class'] = 7

        else:
            raise ValueError(df.loc[index, 'time'] + '找不到`8分类`')

    return df


def create_X_Y(df, seq_length):
    X = []
    Y = []
    for i in range(df.shape[0] - seq_length):
        X.append(np.array(df.iloc[i: i + seq_length, 6:]))
        Y.append(df.loc[i + seq_length, 'nor_cl'])
    X = np.array(X)
    Y = np.array(Y).reshape(len(Y), 1)

    return X, Y


def read_data(building_name, model_name, seq_length, temperature_min, temperature_max):
    # load data
    csv_path = '../../raw_data/'
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

    # config context
    df = context_setting_weather_day(df=df)

    # normalize
    df, load_min, load_max = normalize(df=df, temperature_min=temperature_min, temperature_max=temperature_max)

    # one_hot time feature and weekday feature
    time_feature = []
    weekday_feature = []
    for i in range(df.shape[0]):
        time_feature_i = str(df.loc[i, :]['time']).split(' ')[-1]
        weekday_feature_i = time.strptime(str(df.loc[i, :]['time']), "%Y-%m-%d %H:%M:%S").tm_wday
        time_feature.append(time_feature_i)
        weekday_feature.append(weekday_feature_i)
    df['time_feature'] = np.array(time_feature)
    df['weekday_feature'] = np.array(weekday_feature)
    df = one_hot(df, 'time_feature')
    df = one_hot(df, 'weekday_feature')

    return df, load_max, load_min


if __name__ == '__main__':
    read_data(building_name='OIE',
              model_name='OIE',
              seq_length=24,
              temperature_min=0,
              temperature_max=40)
