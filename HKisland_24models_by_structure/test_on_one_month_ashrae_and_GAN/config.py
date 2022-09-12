temperature_max = 40
temperature_min = 0

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

model_structure_names = ['lstm', 'sparse_lstm', 'sparse_ed', 'seq2seq_with_attention']
building_names = ['CP4', 'DOH', 'OIE']
season_names = ['spring', 'summer']
model_name_list = ['{}_{}_{}'.format(model_structure_name, season_name, building_name)
                   for model_structure_name in model_structure_names
                   for season_name in season_names
                   for building_name in building_names]

# note: change the following parameters
building_name = 'OXH'

source_context = 1
target_context = 4
one_month_start_date = '2017-02-01 00:00:00'
one_month_end_date = '2017-02-28 23:00:00'

# the building where the GAN is trained on
GAN_building_name = 'OIE'
gen_a_cl_save_path = '../../UNIT/models/{}_ashrae_{}_to_{}_cl.pt'.format(GAN_building_name, source_context, target_context)
gen_b_cl_save_path = '../../UNIT/models/{}_ashrae_{}_to_{}_cl.pt'.format(GAN_building_name, target_context, source_context)
