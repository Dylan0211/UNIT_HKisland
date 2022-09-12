dis_learning_rate = 0.0001
gen_learning_rate = 0.001
num_epochs = 200
batch_size = 64
train_size = 0.5

num_data = 60

context_a = 1
context_b = 4

read_data_building_name = 'OXH'
train_building_name = 'OIE'
test_building_name = 'OIE'
gen_a_cl_save_path = './models/{}_ashrae_{}_to_{}_cl.pt'.format(train_building_name, context_a, context_b)
gen_b_cl_save_path = './models/{}_ashrae_{}_to_{}_cl.pt'.format(train_building_name, context_b, context_a)

temperature_min = 0
temperature_max = 40
