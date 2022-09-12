from HKisland_24models_by_structure.dataloader import read_data
from config import *

import numpy as np
import pickle


def create_X_Y(df):
    X = []
    Y = []
    for i in range(df.shape[0] - seq_length):
        X.append(np.array(df.iloc[i: i + seq_length, 6:]))
        Y.append(df.loc[i + seq_length, 'nor_cl'])
    X = np.array(X)
    Y = np.array(Y).reshape(len(Y), 1)

    return X, Y


def load_data():
    # load all data
    for building_name in building_names:
        for season_name in season_names:
            model_name = '{}_{}_{}'.format(model_structure_name, season_name, building_name)
            # load df
            df, load_max, load_min = read_data(building_name=building_name,
                                               model_name=model_name,
                                               seq_length=seq_length,
                                               temperature_min=temperature_min,
                                               temperature_max=temperature_max)

            # get data within specific season
            if season_name == 'spring':
                df = df[((df['time'] >= '2017-03-01 00:00:00') & (df['time'] <= '2017-07-16 23:00:00')) |
                        ((df['time'] >= '2018-03-01 00:00:00') & (df['time'] <= '2018-07-16 23:00:00'))]
            elif season_name == 'summer':
                df = df[((df['time'] >= '2017-04-15 23:00:00') & (df['time'] <= '2017-08-31 23:00:00')) |
                        ((df['time'] >= '2018-04-15 23:00:00') & (df['time'] <= '2018-08-31 23:00:00'))]
            df.reset_index(drop=True, inplace=True)

            # create X and Y
            X, Y = create_X_Y(df=df)

            # save
            print('X shape: {}'.format(X.shape))
            print('Y shape: {}'.format(Y.shape))
            with open('../tmp_pkl_data/{}_X_Y_max_min.pkl'.format(model_name), 'wb') as w:
                pickle.dump((X, Y, load_max, load_min), w)
            print('model saved: ../tmp_pkl_data/{}_X_Y_max_min.pkl'.format(model_name))
            print()


if __name__ == '__main__':
    load_data()
