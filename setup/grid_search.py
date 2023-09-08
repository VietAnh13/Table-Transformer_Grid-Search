import numpy as np

search_space = []
len_search_space = 0

#lr_space = [5e-7, 1e-6, 5e-6, 1e-5, 5e-5]
#lr_space = [1.25e-4, 1.25e-4, 1.25e-4, 1.25e-4]
lr_space = [1.25e-4]
#lr_space = [5e-7]
batch_size_space_td = [2, 4, 8]
#batch_size_space_td = [2]
#batch_size_space_tsr = [2, 4]
batch_size_space_tsr = [2, 2, 2]
batch_size_space = []
epochs_space = [200]
#num_freezing_layer_space = [0, 5, 10]
num_freezing_layer_space = [0]

len_lr_space = 0
len_batch_size_space = 0
len_epochs_space = 0
len_num_freezing_layer_space = 0

max_metrics_in_val = [0, 0, 0, 0]

current_hyper_params_value = []
best_hyper_params_match = []

order_optimize_gs = -1

base_model_save_dir = ''
best_model_name = ''

def get_base_model_save_dir():
    global base_model_save_dir
    return base_model_save_dir

def update_base_model_save_dir(save_dir):
    global base_model_save_dir
    base_model_save_dir = save_dir

def get_best_model_name():
    global best_model_name
    return best_model_name

def update_best_model_name(model_name):
    global best_model_name
    best_model_name = model_name

def update_best_hyper_params_match(hyper_params_arr):
    global best_hyper_params_match
    best_hyper_params_match = hyper_params_arr.copy()

def get_best_hyper_params_match():
    global best_hyper_params_match
    return best_hyper_params_match

def update_order_optimize_gs(order_values):
    global order_optimize_gs
    order_optimize_gs = order_values

def get_current_hyper_params_value():
    global current_hyper_params_value
    return current_hyper_params_value

def update_current_hyper_params_value(hyper_params_arr):
    global current_hyper_params_value
    current_hyper_params_value = hyper_params_arr.copy()

def create_line_hyper_params_to_write(data_hyper_params):
    HYPER_PARAMS_NAME_SHORTCUT = ['lr', 'batch_size', 'epochs', 'num_freezing_layer']
    len_data_hyper_params = len(data_hyper_params)
    line_data_hyper_params = '{'
    for i in range(len(data_hyper_params)):
        line_data_hyper_params += f'\'{HYPER_PARAMS_NAME_SHORTCUT[i]}\': {data_hyper_params[i]}'
        if i < len_data_hyper_params - 1:
            line_data_hyper_params += ', '
        else:
            line_data_hyper_params += '}'

    return line_data_hyper_params

def create_line_data_to_write(data_arr):
    space_separator = ' '
    str_data_arr = []
    for i in range(len(data_arr)):
        str_data_arr.append('{:.3f}'.format(data_arr[i]))
    line_data = space_separator.join(str_data_arr)
    return line_data

def write_data_to_file_txt(file_name, save_dir_folder, data_write, first_line):
    if first_line:
        with open(f'{save_dir_folder}/{file_name}.txt', 'w') as f:
            f.write("%s\n" % data_write)
    else:
        with open(f'{save_dir_folder}/{file_name}.txt', 'a') as f:
            f.write("%s\n" % data_write)

def update_length_learning_rate_space():
    global len_lr_space
    len_lr_space = len(lr_space)

def update_length_and_batch_size_space(train_mode):
    global batch_size_space
    global len_batch_size_space
    if train_mode == 'TD':
        batch_size_space = batch_size_space_td.copy()
    elif train_mode == 'TSR':
        batch_size_space = batch_size_space_tsr.copy()
    len_batch_size_space = len(batch_size_space)

def update_length_epochs_space():
    global len_epochs_space
    len_epochs_space = len(epochs_space)

def update_length_num_freezing_layer_space():
    global len_num_freezing_layer_space
    len_num_freezing_layer_influence = len (num_freezing_layer_space)

def update_length_search_space():
    global len_search_space
    len_search_space = len(search_space)

    return len_search_space

def create_search_space(train_mode):
    global search_space
    global lr_space
    global batch_size_space
    global epochs_space
    global num_freezing_layer_space

    update_length_learning_rate_space()
    update_length_and_batch_size_space(train_mode)
    update_length_epochs_space()
    update_length_num_freezing_layer_space()

    search_space = [[i, j, k, l] for i in lr_space for j in batch_size_space for k in epochs_space for l in num_freezing_layer_space]
    return update_length_search_space()

def get_hyper_params_in_search_space(index):
    global search_space
    return (search_space[index][key] for key in range(4))

def create_metrics_values(AP50_VALUES, AP75_VALUES, AP_VALUES, AR_VALUES):
    return [AP50_VALUES, AP75_VALUES, AP_VALUES, AR_VALUES]

def fragment_metrics_values(metrics_value):
    data_metrics = metrics_value.copy()
    return data_metrics[0], data_metrics[1], data_metrics[2], data_metrics[3]

def get_max_metrics_in_val():
    global max_metrics_in_val
    return max_metrics_in_val

def update_metrics_value(metrics_values, values):
    metrics_values[:] = values.copy()

def check_update_max_metrics_values(current_metrics_values, max_metrics_values):
    current_AP50, current_AP75, current_AP, current_AR = fragment_metrics_values(current_metrics_values)
    max_AP50, max_AP75, max_AP, max_AR = fragment_metrics_values(max_metrics_values)
    if current_AP75 > max_AP75:
        return True
    elif current_AP75 == max_AP75:
        if current_AP > max_AP:
            return True
        elif current_AP == max_AP:
            if current_AP50 > max_AP50:
                return True
            elif current_AP50 == max_AP50:
                if current_AR > max_AR:
                    return True
    return False

def update_max_metrics_values(current_metrics_values, max_metrics_values, order_values):
    if check_update_max_metrics_values(current_metrics_values, max_metrics_values):
        update_metrics_value(max_metrics_values, current_metrics_values)
        update_order_optimize_gs(order_values)
        update_best_hyper_params_match(get_current_hyper_params_value())