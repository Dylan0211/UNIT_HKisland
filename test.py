from model import Generator, TrainSet
from config import *
from torch.utils.data import DataLoader

import torch
import pickle
import matplotlib.pyplot as plt


def UNIT_Test(test_building_name):
    # set up models
    gen_a = Generator()
    gen_b = Generator()
    gen_a.load_state_dict(torch.load(gen_a_save_path))
    gen_b.load_state_dict(torch.load(gen_b_save_path))

    # encode and decode (from style a to b)
    encode = gen_a.encode
    decode = gen_b.decode

    # load data and set data loader
    with open('./tmp_pkl_data/{}_a_b_max_min_context.pkl'.format(test_building_name), 'rb') as r:
        data_a, data_b, load_max, load_min, df = pickle.load(r)
    # data_a = data_a[int(train_size * data_a.shape[0]):]
    # data_b = data_b[int(train_size * data_b.shape[0]):]
    test_dataset = TrainSet(data_a, data_b)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    print('test on: {}.csv'.format(test_building_name))
    print('data_a shape: {}'.format(data_a.shape))
    print('data_b shape: {}'.format(data_b.shape))

    # start testing
    real = []
    fake = []
    original = []
    for x_a, x_b in test_loader:
        x_a = x_a.to(torch.float32)
        content, _ = encode(x_a)
        output = decode(content)
        real.append(x_b.detach().numpy())
        fake.append(output.detach().numpy())
        original.append(x_a.detach().numpy())

    # data processing
    real_data = []
    fake_data = []
    original_data = []
    for i in range(len(real)):
        for j in range(real[i].shape[0]):
            real_data.append(real[i][j])
            fake_data.append(fake[i][j])
            original_data.append(original[i][j])

    # remove redundant data
    i = 0
    unique_real_data = []
    unique_fake_data = []
    unique_original_data = []
    while i < len(real_data):
        for j in range(len(real_data[i])):
            unique_real_data.append(real_data[i // 50][j])
            unique_fake_data.append(fake_data[i][j])
            unique_original_data.append(original_data[i][j])
        i += num_data

    # denormalize
    final_real_data = [unique_real_data[i] * (load_max - load_min) + load_min
                       for i in range(len(unique_real_data))]
    final_fake_data = [unique_fake_data[i] * (load_max - load_min) + load_min
                       for i in range(len(unique_fake_data))]
    final_original_data = [unique_original_data[i] * (load_max - load_min) + load_min
                           for i in range(len(unique_original_data))]

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
    plt.title(test_building_name)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    UNIT_Test(test_building_name=test_building_name)








