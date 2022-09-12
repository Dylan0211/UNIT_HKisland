temperature_max = 40
temperature_min = 0

building_name = 'OIE'  # note: choose building to train on
season_name = 'summer'  # note: choose season (spring & summer)
model_structure_name = 'lstm'  # note: choose model structure

model_name = '{}_{}_{}'.format(model_structure_name, season_name, building_name)

seq_length = 24
input_dim = 33
sparse_output_dim = 64
hidden_dim = 128
output_dim = 1
num_layers = 1

enc_hid_dim = 128
dec_hid_dim = 128
dropout = 0.5

sparse_lstm_lambda = 1e-3
sparse_ed_lambda = 1e-5

train_ratio = 0.5
batch_size = 64
learning_rate = 0.001
num_epochs = 10
