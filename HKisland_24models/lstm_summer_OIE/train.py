from config import *
from torch.utils.data import DataLoader
from HKisland_24models.model import *

import pickle


def train_model():
    # load data
    with open('../tmp_pkl_data/{}_X_Y_max_min.pkl'.format(model_name), 'rb') as r:
        data_x, data_y, load_max, load_min = pickle.load(r)
    data_x = data_x[:int(len(data_x) * train_ratio)]
    data_y = data_y[:int(len(data_y) * train_ratio)]
    print('X shape: {}'.format(data_x.shape))
    print('Y shape: {}'.format(data_y.shape))
    print('load max: {}'.format(load_max))
    print('load min: {}'.format(load_min))

    # set train set and dataloader
    train_set = TrainSet(data_x, data_y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # set up trained_models
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
    criterion = nn.L1Loss()
    model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    for epoch in range(num_epochs):
        l_sum = 0.
        for data_x, data_y in train_loader:
            data_x = data_x.to(torch.float32)
            data_y = data_y.to(torch.float32)

            pred = model(data_x)
            loss = criterion(pred, data_y)

            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            l_sum += loss.item()

        print('Epoch {}: loss = {}'.format(epoch + 1, l_sum / len(train_loader)))

    # save model
    save_path = '../trained_models/{}.pt'.format(model_name)
    torch.save(model.state_dict(), save_path)
    print('model saved: {}'.format(save_path))


if __name__ == '__main__':
    train_model()
