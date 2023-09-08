from visualize import *
from grid_search import *

import os

# RUN_MODE = 0 <default mode>
# RUN_MODE = 1 <visualize mode>
# RUN_MODE = 2 <grid search mode>

RUN_MODE = -1
NUM_OF_RUN_MODE = 3
TRAIN_TYPE = 'TD'
ORDER_TRAIN_EVAL = ''

current_order = -1
standard_deviation_order = -1
first_update_standard_deviation_order = True

LAST_MODEL_SAVE_DIR = ''
finish_epochs_progress = True

with open('../setup/setup.txt', 'r') as f:
    lines = f.readlines()
    RUN_MODE = int(lines[0])
    ORDER_TRAIN_EVAL = str(lines[3]).replace('\n', '')
    TRAIN_TYPE = str(lines[5]).replace('\n', '')

RUN_MODE_STATUS = [True, False, False]
len_of_run_mode_status = len(RUN_MODE_STATUS)
len_of_run_mode = [-1, -1, -1]

highest_metrics_epochs_val_accuracy = [0, 0, 0, 0]
epochs_highest_metrics = 0

run_mode_name = ['default_mode', 'visualize_mode', 'grid_search_mode']

def update_progress_status():
    global LAST_MODEL_SAVE_DIR
    global finish_epochs_progress
    global RUN_MODE
    global TRAIN_TYPE
    global ORDER_TRAIN_EVAL
    global first_update_standard_deviation_order
    global RUN_MODE_STATUS

    file_name = f'log_crash_{ORDER_TRAIN_EVAL}'
    save_dir_folder = f'../log_crash/{get_run_mode_name()[RUN_MODE]}/{TRAIN_TYPE}/{ORDER_TRAIN_EVAL}'

    if os.path.isfile(f'{save_dir_folder}/{file_name}.txt'):
        with open(f'{save_dir_folder}/{file_name}.txt') as f:
            lines = f.readlines()
            if len(lines) > 0:
                if str(lines[0]).replace('\n', '') == 'False':
                    if str(lines[2]).replace('\n', '') == 'True':
                        finish_epochs_progress = True
                        if first_update_standard_deviation_order:
                            update_standard_deviation_order(int(lines[1]) + 1)
                            first_update_standard_deviation_order = False
                    elif str(lines[2]).replace('\n', '') == 'False':
                        finish_epochs_progress = False
                        if first_update_standard_deviation_order:
                            update_standard_deviation_order(int(lines[1]))
                            LAST_MODEL_SAVE_DIR = str(lines[3]).replace('\n', '')
                            first_update_standard_deviation_order = False
            else:
                finish_epochs_progress = True
                if first_update_standard_deviation_order:
                    update_standard_deviation_order(0)
                    first_update_standard_deviation_order = False
    else:
        finish_epochs_progress = True
        if first_update_standard_deviation_order:
            update_standard_deviation_order(0)
            first_update_standard_deviation_order = False
    
    if get_standard_deviation_order() >= 0:
        if RUN_MODE_STATUS[0]:
            current_index = get_current_order() + get_standard_deviation_order()
            file_name_MVAHP = f'metrics_val_accuracy_hyper_params_{current_index + 1}'
            save_dir_MVAHP = f'../run_mode_log/grid_search_mode/{TRAIN_TYPE}/{ORDER_TRAIN_EVAL}'
            if os.path.isfile(f'{file_name_MVAHP}/{save_dir_MVAHP}.txt'):
                 with open(f'{save_dir_MVAHP}/{file_name_MVAHP}.txt') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        epoch = -1
                        for line in lines:
                            epoch += 1
                            line_metrics = str(line).replace('\n', '')
                            current_line_metrics = line_metrics.split()
                            AP50_current = float(current_line_metrics[0])
                            AP75_current = float(current_line_metrics[1])
                            AP_current = float(current_line_metrics[2])
                            AR_current = float(current_line_metrics[3])
                            current_metrics = create_metrics_values(AP50_current, AP75_current, AP_current, AR_current)
                            update_highest_metrics_epochs_val_accuracy(current_metrics, epoch + 1)

        if RUN_MODE_STATUS[2]:
            file_name_HDMC = f'highest_metrics_val_accuracy_current'
            save_dir_HDMC = f'../run_mode_log/grid_search_mode/{TRAIN_TYPE}/{ORDER_TRAIN_EVAL}'
            if os.path.isfile(f'{save_dir_HDMC}/{file_name_HDMC}.txt'):
                with open(f'{save_dir_HDMC}/{file_name_HDMC}.txt') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        line_highest_metrics = str(lines[0]).replace('\n', '')
                        highest_metrics = line_highest_metrics.split()
                        AP50_HM = float(highest_metrics[0])
                        AP75_HM = float(highest_metrics[1])
                        AP_HM = float(highest_metrics[2])
                        AR_HM = float(highest_metrics[3])
                        current_highest_metrics = create_metrics_values(AP50_HM, AP75_HM, AP_HM, AR_HM)
                        update_metrics_value(get_max_metrics_in_val(), current_highest_metrics)
                        HYPER_PARAMS_NAME_SHORTCUT = ['lr', 'batch_size', 'epochs', 'num_freezing_layer']
                        dict_best_hyper_params = eval('{' + line_highest_metrics.split('{')[1].split('}')[0] + '}')
                        lr_HM = dict_best_hyper_params[HYPER_PARAMS_NAME_SHORTCUT[0]]
                        bs_HM = dict_best_hyper_params[HYPER_PARAMS_NAME_SHORTCUT[1]]
                        ep_HM = dict_best_hyper_params[HYPER_PARAMS_NAME_SHORTCUT[2]]
                        nf_HM = dict_best_hyper_params[HYPER_PARAMS_NAME_SHORTCUT[3]]
                        current_best_hyper_params = create_metrics_values(lr_HM, bs_HM, ep_HM, nf_HM)
                        update_best_hyper_params_match(current_best_hyper_params)

def get_run_mode_name():
    global run_mode_name
    return run_mode_name

def get_epochs_highest_metrics():
    global epochs_highest_metrics
    return epochs_highest_metrics

def update_epochs_highest_metrics(epoch):
    global epochs_highest_metrics
    epochs_highest_metrics = epoch

def reset_highest_metrics_epochs_val_accuracy():
    global highest_metrics_epochs_val_accuracy
    highest_metrics_epochs_val_accuracy = [0, 0, 0 ,0]

def get_highest_metrics_epochs_val_accuracy():
    global highest_metrics_epochs_val_accuracy
    return highest_metrics_epochs_val_accuracy

def update_highest_metrics_epochs_val_accuracy(metrics_epochs_val_accuracy, epoch):
    global highest_metrics_epochs_val_accuracy
    if check_update_max_metrics_values(metrics_epochs_val_accuracy, highest_metrics_epochs_val_accuracy):
        highest_metrics_epochs_val_accuracy = metrics_epochs_val_accuracy.copy()
        update_epochs_highest_metrics(epoch)

def get_run_mode():
    global RUN_MODE
    return RUN_MODE

def get_order_train_eval():
    global ORDER_TRAIN_EVAL
    return ORDER_TRAIN_EVAL

def get_finish_epochs_progress():
    global finish_epochs_progress
    return finish_epochs_progress

def get_last_model_save_dir():
    global LAST_MODEL_SAVE_DIR
    return LAST_MODEL_SAVE_DIR

def update_current_order(index):
    global current_order
    current_order = index

def get_current_order():
    global current_order
    return current_order

def update_standard_deviation_order(index):
    global standard_deviation_order
    standard_deviation_order = index

def get_standard_deviation_order():
    global standard_deviation_order
    return standard_deviation_order

def write_state_run_mode(data_write):
    global RUN_MODE
    global TRAIN_TYPE
    global ORDER_TRAIN_EVAL
    file_name = f'log_crash_{ORDER_TRAIN_EVAL}'
    save_dir_folder = f'../log_crash/{get_run_mode_name()[RUN_MODE]}/{TRAIN_TYPE}/{ORDER_TRAIN_EVAL}'
    os.makedirs(save_dir_folder, exist_ok=True)

    with open(f'{save_dir_folder}/{file_name}.txt', 'w') as f:
        f.write("%s\n" % data_write)

def write_weights_save_dir(weights_save_dir, data_write):
    file_name = f'weights_save_dir'

    with open(f'{weights_save_dir}/{file_name}.txt', 'w') as f:
        f.write("%s\n" % data_write)

def update_train_type():
    global TRAIN_TYPE
    return TRAIN_TYPE

def update_mode_run():
    reset_arr(RUN_MODE_STATUS)
    for i in range(NUM_OF_RUN_MODE):
        if i == RUN_MODE:
            RUN_MODE_STATUS.append(True)
        else:
            RUN_MODE_STATUS.append(False)
    
    global len_of_run_mode_status
    len_of_run_mode_status = len(RUN_MODE_STATUS)

    update_progress_status()
    global standard_deviation_order
    if standard_deviation_order > 0:
        denta_len_run_mode = standard_deviation_order
    else:
        denta_len_run_mode = 0
    
    if RUN_MODE == 0:
        len_of_run_mode[0] = 1 - denta_len_run_mode
    elif RUN_MODE == 1:
        len_of_run_mode[RUN_MODE] = update_batch_size_influence(TRAIN_TYPE) - denta_len_run_mode
    elif RUN_MODE == 2:
        len_of_run_mode[RUN_MODE] = create_search_space(TRAIN_TYPE) - denta_len_run_mode