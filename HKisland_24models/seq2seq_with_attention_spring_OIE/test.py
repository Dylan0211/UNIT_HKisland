"""
test的数据暂时是第二年的夏天，此处应改用生成的数据test
"""
from config import *
from torch.utils.data import DataLoader
from HKisland_24models.model import *

import pickle
import torch
import numpy as np
import math
import matplotlib.pyplot as plt


def draw_mse_and_rmse(label_list, pred_list):
    mae_list = [abs(label_list[i] - pred_list[i]) for i in range(len(pred_list))]
    mae = sum(mae_list) / len(mae_list)
    rmse_list = [(label_list[i] - pred_list[i]) ** 2 for i in range(len(pred_list))]
    rmse = math.sqrt(sum(rmse_list) / len(rmse_list))

    fig = plt.figure(figsize=(10, 6))
    fig.add_subplot(111)
    plt.plot(range(len(pred_list)), pred_list, marker='8', color='orange', linewidth=1, label='predict')
    plt.plot(range(len(label_list)), label_list, color='b', linewidth=1, label='label')
    title = 'MAE={:.3f} \n RMSE={:.3f}'.format(mae, rmse)
    plt.title('Model output vs actual')
    plt.title(title, loc='right')
    plt.legend()
    plt.show()


def test_model():
    # GPU info
    print('*' * 40)
    print('GPU information are as follows:')
    if_cuda = torch.cuda.is_available()
    print('if_cuda =', if_cuda)
    print('*' * 40)
    print('Gpu_count =', torch.cuda.device_count())
    print('*' * 40)
    print('gpu_name =', torch.cuda.get_device_name(0))
    print('*' * 40)

    device = torch.device("cuda:0" if if_cuda else "cpu")
    print("Set program to device:", torch.cuda.get_device_name(device))
    print('*' * 40)

    # load data
    with open('../tmp_pkl_data/{}_X_Y_max_min.pkl'.format(model_name), 'rb') as r:
        data_x, data_y, load_max, load_min = pickle.load(r)
    data_x = data_x[int(len(data_x) * train_ratio):]
    data_y = data_y[int(len(data_y) * train_ratio):]
    print('X shape: {}'.format(data_x.shape))
    print('Y shape: {}'.format(data_y.shape))
    print('load max: {}'.format(load_max))
    print('load min: {}'.format(load_min))

    # set test set and loader
    test_set = TrainSet(data_x, data_y)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # set up model
    model_dict = {
        'lstm': LSTM(input_dim=input_dim,
                     hidden_dim=hidden_dim,
                     num_layers=num_layers),
        'sparse_lstm': Sparse_LSTM(input_dim=input_dim,
                                   hidden_dim=hidden_dim,
                                   num_layers=num_layers,
                                   sparse_output_dim=sparse_output_dim,
                                   output_dim=output_dim),
        'sparse_ed': Seq2Seq(input_dim=input_dim,
                             hidden_dim=hidden_dim,
                             num_layers=num_layers,
                             sparse_output_dim=sparse_output_dim,
                             output_dim=output_dim),
        'seq2seq_with_attention': Seq2Seq_with_attention(input_dim=input_dim,
                                                         output_dim=output_dim,
                                                         enc_hid_dim=enc_hid_dim,
                                                         dec_hid_dim=dec_hid_dim,
                                                         dropout=dropout),
    }
    model = model_dict.get(model_structure_name)
    save_path = '../trained_models/{}.pt'.format(model_name)
    model.load_state_dict(torch.load(save_path))
    model = model.to(device)

    # start testing
    preds = []
    for data_x, _ in test_loader:
        data_x = data_x.to(torch.float32)
        data_x = data_x.to(device)

        pred = model(data_x)
        pred = pred.cpu().detach().numpy()
        preds.append(pred)

    # denormalize
    _label_list = test_set.label.tolist()
    _pred_list = []
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            _pred_list.append(preds[i][j])
    label_list = []
    pred_list = []
    error_list = []
    for k in range(len(_pred_list)):
        label_list.append(_label_list[k][0] * (load_max - load_min) + load_min)
        pred_list.append(_pred_list[k][0] * (load_max - load_min) + load_min)
        error_list.append(abs(label_list[k] - pred_list[k]))

    # print inference results
    square_error_list = [(label_list[i] - pred_list[i]) ** 2 for i in range(len(pred_list))]
    print('!! MAE =', np.array(error_list).mean())
    print('!! RMSE =', math.sqrt(sum(square_error_list) / len(square_error_list)))

    # draw graphs
    draw_mse_and_rmse(label_list, pred_list)

    # save prediction results
    # with open('../prediction_result/{}_prediction.pkl'.format(model_name), 'wb') as w:
    #     pickle.dump((label_list, pred_list, error_list), w)
    # print('prediction result saved: {}_prediction.pkl'.format(model_name))


if __name__ == '__main__':
    test_model()
