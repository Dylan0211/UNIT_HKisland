from GAN_model import Generator, TrainSet
from config import *
from torch.utils.data import DataLoader

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np


def generate_data(df):
    # set up trained_models
    gen_a = Generator()
    gen_b = Generator()
    gen_a.load_state_dict(torch.load(gen_a_cl_save_path))
    gen_b.load_state_dict(torch.load(gen_b_cl_save_path))

    # get domain a and domain b data
    df_context_a = df[df['8_class'] == context_a]
    df_context_b = df[df['8_class'] == context_b]
    df_context_a.reset_index(drop=True, inplace=True)
    df_context_b.reset_index(drop=True, inplace=True)

    temp_a = df_context_a[:24 * num_data]
    temp_b = df_context_b[:24 * num_data]

    data_a = []
    data_b = []
    for i in range(num_data):
        data_a.append(temp_a.loc[24 * i: 24 * (i + 1) - 1, 'nor_cl'])
        data_b.append(temp_b.loc[24 * i: 24 * (i + 1) - 1, 'nor_cl'])
    data_a = np.array(data_a)
    data_b = np.array(data_b)

    test_dataset = TrainSet(data_a, data_b)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    print('test on: {}.csv'.format(test_building_name))
    print('data_a shape: {}'.format(data_a.shape))
    print('data_b shape: {}'.format(data_b.shape))

    # start testing
    fake = []
    real = []
    original = []
    for x_a, x_b in test_loader:
        x_a = x_a.to(torch.float32)
        content, _ = gen_a.encode(x_a)
        output = gen_b.decode(content)
        fake.append(output.detach().numpy())
        real.append(x_b.detach().numpy())
        original.append(x_a.detach().numpy())

    # data processing
    fake_data = []
    real_data = []
    original_data = []
    for i in range(len(fake)):
        for j in range(fake[i].shape[0]):
            fake_data.append(fake[i][j])
            real_data.append(real[i][j])
            original_data.append(original[i][j])

    final_fake_data = []
    final_real_data = []
    final_original_data = []
    for i in range(len(real_data)):
        for j in range(real_data[i].shape[0]):
            final_real_data.append(real_data[i][j])
            final_fake_data.append(fake_data[i][j])
            final_original_data.append(original_data[i][j])

    return final_fake_data, final_real_data, final_original_data


def UNIT_Test_cl(test_building_name):
    # load data and set data loader
    with open('./tmp_pkl_data/{}_ashrae_{}_to_{}_cl.pkl'.format(test_building_name, context_a, context_b), 'rb') as r:
        _, _, load_max, load_min, df = pickle.load(r)

    fake_data, real_data, original_data = generate_data(df=df)

    # denormalize
    final_fake_data = [fake_data[i] * (load_max - load_min) + load_min for i in range(len(fake_data))]
    final_real_data = [real_data[i] * (load_max - load_min) + load_min for i in range(len(real_data))]
    final_original_data = [original_data[i] * (load_max - load_min) + load_min for i in range(len(original_data))]

    # error
    mae_list = [abs(final_real_data[i] - final_fake_data[i]) for i in range(len(final_real_data))]
    mae = sum(mae_list) / len(mae_list)

    # draw graphs
    fig = plt.figure(figsize=(10, 6))
    fig.add_subplot(111)
    plt.plot(range(len(final_real_data)), final_real_data, label='real_data', color='blue')
    plt.plot(range(len(final_fake_data)), final_fake_data, label='fake_data', color='orange')
    plt.plot(range(len(final_original_data)), final_original_data, label='original_data', color='green')
    plt.title('MAE = {}'.format(mae), loc='right')
    plt.title('{}_coolingLoad'.format(test_building_name))
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    UNIT_Test_cl(test_building_name=test_building_name)

