dis_learning_rate = 0.0001
gen_learning_rate = 0.001
num_epochs = 200
batch_size = 64
train_size = 0.5

temperature_min = 0
temperature_max = 40

# note: change these parameters to change data to be loaded, trained or tested
# if context_a and context_b are both weekdays, use dataloader_new_weekday_to_weekday.py to load
# if context_a is weekday and context_b is weekend, use dataloader_new_weekday_to_weekend.py to load
"""
winter peak weekday,                0
winter average weekday,             1
winter average weekend day/holiday, 2

summer peak weekday,                3
summer average weekday,             4
summer average weekend day/holiday, 5

spring average weekday,             6 
fall average weekday                7
"""
context_a = 2
context_b = 5
context_a_is_weekend = True if context_a in [2, 5] else False
context_b_is_weekend = True if context_b in [2, 5] else False

read_data_building_name = 'OXH'
train_building_name = 'OIE'
test_building_name = 'LIH'
gen_a_cl_save_path = './models/{}_ashrae_{}_to_{}_cl.pt'.format(train_building_name, context_a, context_b)
gen_b_cl_save_path = './models/{}_ashrae_{}_to_{}_cl.pt'.format(train_building_name, context_b, context_a)

