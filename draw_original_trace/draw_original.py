import pickle
import matplotlib.pyplot as plt

from config import *


def draw_gt_trace():
    # load data
    csv_path = './tmp_pkl_data/{}_max_min_context.pkl'.format(building_name)
    with open(csv_path, 'rb') as r:
        load_max, load_min, df = pickle.load(r)
    load_gt = df.loc[:, 'coolingLoad'].tolist()

    # draw trace
    fig = plt.figure(figsize=(10, 6))
    fig.add_subplot(111)
    plt.plot(range(len(load_gt)), load_gt, color='blue')
    plt.grid()
    plt.title(building_name)
    plt.show()


if __name__ == '__main__':
    draw_gt_trace()
