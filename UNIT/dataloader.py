"""

"""
from config import *
from datetime import datetime
from make_cl_sum import *

import pickle
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


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


def create_a_and_b_weekday(df, context_a, context_b):
    # get domain a and domain b data
    df_context_a = df[df['8_class'] == context_a]
    df_context_b = df[df['8_class'] == context_b]
    df_context_a.reset_index(drop=True, inplace=True)
    df_context_b.reset_index(drop=True, inplace=True)

    # check data a and b
    check_a = df_context_a.loc[:24 * num_data - 1, 'nor_cl'].tolist()
    check_b = df_context_b.loc[:24 * num_data - 1, 'nor_cl'].tolist()
    fig = plt.figure(figsize=(10, 6))
    fig.add_subplot(111)
    plt.plot(range(len(check_a)), check_a, color='orange', label='domain_a')
    plt.plot(range(len(check_b)), check_b, color='blue', label='domain_b')
    plt.title(read_data_building_name)
    plt.legend()
    plt.grid()
    # plt.show()

    # get each weekday data
    df_context_a = df_context_a[:24 * num_data]
    df_context_b = df_context_b[:24 * num_data]
    context_a_each_weekday_data = [[] for i in range(5)]
    context_b_each_weekday_data = [[] for i in range(5)]
    for i in range(df_context_a.shape[0]):
        context_a_each_weekday_data[df_context_a.loc[i, 'weekday_feature']].append(df_context_a.loc[i, 'time'])
        context_b_each_weekday_data[df_context_b.loc[i, 'weekday_feature']].append(df_context_b.loc[i, 'time'])

    # get each time data and mappings
    cl_a = []
    cl_b = []
    for i in range(len(context_a_each_weekday_data)):  # every weekday data
        # get each time data
        # note: 潜在问题，要是取最后的数据可能不足24h
        context_a_each_time_data = [[] for i in range(24)]  # every hour data
        for j in range(len(context_a_each_weekday_data[i])):
            this_time = time.strptime(str(context_a_each_weekday_data[i][j]), "%Y-%m-%d %H:%M:%S").tm_hour
            this_index = df[df['time'] == context_a_each_weekday_data[i][j]].index[0]
            this_24h_cl = np.array(df.loc[this_index: this_index + 23, 'nor_cl'])
            context_a_each_time_data[this_time].append(this_24h_cl)
        context_b_each_time_data = [[] for i in range(24)]
        for j in range(len(context_b_each_weekday_data[i])):
            this_time = time.strptime(str(context_b_each_weekday_data[i][j]), "%Y-%m-%d %H:%M:%S").tm_hour
            this_index = df[df['time'] == context_b_each_weekday_data[i][j]].index[0]
            this_24h_cl = np.array(df.loc[this_index: this_index + 23, 'nor_cl'])
            context_b_each_time_data[this_time].append(this_24h_cl)

        # get mappings
        for j in range(len(context_a_each_time_data)):  # each hour
            for k in range(len(context_a_each_time_data[j])):  # context a data
                for l in range(len(context_b_each_time_data[j])):  # context b data
                    temp_a = context_a_each_time_data[j][k].shape[0]
                    temp_b = context_b_each_time_data[j][l].shape[0]
                    if temp_a == 23 or temp_b == 0:
                        aaa = 1
                    cl_a.append(context_a_each_time_data[j][k])
                    cl_b.append(context_b_each_time_data[j][l])

    cl_a = np.array(cl_a)
    cl_b = np.array(cl_b)

    return cl_a, cl_b


def one_hot(data, feature):
    onehot = pd.get_dummies(data[feature])
    df = data.drop(feature, axis=1)
    data = df.join(onehot)
    return data


def read_data():
    # load data
    csv_path = '../raw_data/'
    building_csv_path = csv_path + read_data_building_name + '.csv'
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

    # config context
    df = context_setting_weather_day(df=df)

    # add weekday feature
    weekday_feature = []
    for i in range(df.shape[0]):
        weekday_feature_i = time.strptime(str(df.loc[i, :]['time']), "%Y-%m-%d %H:%M:%S").tm_wday
        weekday_feature.append(weekday_feature_i)
    df['weekday_feature'] = np.array(weekday_feature)

    # create a and b
    cl_a, cl_b = create_a_and_b_weekday(df=df, context_a=context_a, context_b=context_b)

    # save
    print('cl_a shape: {}'.format(cl_a.shape))
    print('cl_b shape: {}'.format(cl_b.shape))
    with open('./tmp_pkl_data/{}_{}_{}_cl.pkl'.format(read_data_building_name, context_a_name, context_b_name), 'wb') as w:
        pickle.dump((cl_a, cl_b, load_max, load_min, df), w)
    print('data saved at: ./tmp_pkl_data/{}_{}_{}_cl.pkl'.format(read_data_building_name, context_a_name, context_b_name))


if __name__ == '__main__':
    read_data()
