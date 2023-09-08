import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from matplotlib.font_manager import FontProperties

HYPERPARAMETER_CHANGE = 0
HYPERPARAMETER_RANGE = []
HYPERPARAMETER_RANGE_START = 3
HYPERPARAMETER_RANGE_STOP = 3

FIGURE_NAME_SHORTCUT = ['LR', 'BS', 'EP', 'NF']
HYPER_PARAMS_NAME_SHORTCUT = ['lr', 'batch_size', 'epochs', 'num_freezing_layer']
HYPER_PARAMS_NAME_FULL = ['LEARNING RATES', 'BATCH SIZES', 'EPOCHS', 'NUM FREEZING LAYERS']

#lr_influence = [5e-7, 5e-6, 5e-5, 5e-4, 5e-3]
lr_influence = [1e-4]
#batch_size_influence_td = [2, 4, 8]
batch_size_influence_td = [2]
#batch_size_influence_tsr = [2, 4]
batch_size_influence_tsr = [2]
batch_size_influence = []
epochs_influence = [40]
#num_freezing_layer_influence = [0, 5, 10, 15, 20]
num_freezing_layer_influence = [0]

len_lr_influence = len(lr_influence)
len_batch_size_influence = len(batch_size_influence)
len_epochs_influence = len(epochs_influence)
len_num_freezing_layer_influence = len(num_freezing_layer_influence)

hyperparameters_space = [lr_influence, batch_size_influence, epochs_influence, num_freezing_layer_influence]

base_hyperparameters = [1e-4, 2, 40, 0]
#base_hyperparameters = [5e-5, 2, 20, 5]
base_hyperparameters_activate = [False, True, True, True]
len_base_hyperparameters = len(base_hyperparameters)

len_hyperparameters = [len_lr_influence, len_batch_size_influence, len_epochs_influence, len_num_freezing_layer_influence]

# (01) train_loss & val_loss per epoch
train_loss_per_epoch = []
val_loss_per_epoch = []

# (02) mean train_loss & val_loss per epoch
mean_train_loss_per_epoch = []
mean_val_loss_per_epoch = []

# (03) train_ap & val_ap per epoch
train_ap_per_epoch = []
val_ap_per_epoch = []

# (04) train_ap75 & val_ap75 per epoch
train_ap75_per_epoch = []
val_ap75_per_epoch = []

# (05) train_ap50 & val_ap50 per epoch
train_ap50_per_epoch = []
val_ap50_per_epoch = []

# (06) train_ar & val_ar per epoch
train_ar_per_epoch = []
val_ar_per_epoch = []

# arr epoch [start_epoch, ..., epochs]
arr_epochs = []

# (07) train_lost & val_lost list
train_loss_per_epoch_list = []
val_loss_per_epoch_list = []

# (08) mean train_lost & val_lost list
mean_train_loss_per_epoch_list = []
mean_val_loss_per_epoch_list = []

# (09) train_ap & val_ap list
train_ap_per_epoch_list = []
val_ap_per_epoch_list = []

# (10) train_ap75 & val_ap75 list
train_ap75_per_epoch_list = []
val_ap75_per_epoch_list = []

# (11) train_ap50 & val_ap50 list
train_ap50_per_epoch_list = []
val_ap50_per_epoch_list = []

# (12) train_ar & val_ar list
train_ar_per_epoch_list = []
val_ar_per_epoch_list = []

def update_hyperparameter_change(index):
    global HYPERPARAMETER_CHANGE
    HYPERPARAMETER_CHANGE = HYPERPARAMETER_RANGE[index]

    set_base_hyperparameters_activate(HYPERPARAMETER_CHANGE)

    return HYPERPARAMETER_CHANGE

def update_batch_size_influence(train_type):
    if train_type == 'TD':
        batch_size_influence = batch_size_influence_td.copy()
    elif train_type == 'TSR':
        batch_size_influence = batch_size_influence_tsr.copy()
    global len_batch_size_influence
    len_batch_size_influence = len(batch_size_influence)
    hyperparameters_space[1] = batch_size_influence
    len_hyperparameters[1] = len_batch_size_influence

    for i in range(HYPERPARAMETER_RANGE_START, HYPERPARAMETER_RANGE_STOP + 1, 1):
        for j in range(len_hyperparameters[i]):
            HYPERPARAMETER_RANGE.append(i)
    
    return len(HYPERPARAMETER_RANGE)

    
def set_base_hyperparameters_activate(num_activate):
    for i in range(len_base_hyperparameters):
        if i == num_activate:
            base_hyperparameters_activate[i] = True
        else:
            base_hyperparameters_activate[i] = False

def reset_arr(arr_input):
    arr_input[:] = []

def reset_all_arr():
    # (01) train_loss & val_loss per epoch
    train_loss_per_epoch[:] = []
    val_loss_per_epoch[:] = []
    
    # (02) mean train_lost & val_lost per epoch
    mean_train_loss_per_epoch[:] = []
    mean_val_loss_per_epoch[:] = []   

    # (03) train_ap & val_ap per epoch
    train_ap_per_epoch[:] = []
    val_ap_per_epoch[:] = []

    # (04) train_ap75 & val_ap75 per epoch
    train_ap75_per_epoch[:] = []
    val_ap75_per_epoch[:] = []

    # (05) train_ap50 & val_ap50 per epoch
    train_ap50_per_epoch[:] = []
    val_ap50_per_epoch[:] = []

    # (06) train_ar & val_ar per epoch
    train_ar_per_epoch[:] = []
    val_ar_per_epoch[:] = []

    # (07) train_lost & val_lost list
    train_loss_per_epoch_list[:] = []
    val_loss_per_epoch_list[:] = []

    # (08) mean train_lost & val_lost list
    mean_train_loss_per_epoch_list[:] = []
    mean_val_loss_per_epoch_list[:] = []

    # (09) train_ap & val_ap list
    train_ap_per_epoch_list[:] = []
    val_ap_per_epoch_list[:] = []

    # (10) train_ap75 & val_ap75 list
    train_ap75_per_epoch_list[:] = []
    val_ap75_per_epoch_list[:] = []

    # (11) train_ap50 & val_ap50 list
    train_ap50_per_epoch_list[:] = []
    val_ap50_per_epoch_list[:] = []

    # (12) train_ar & val_ar list
    train_ar_per_epoch_list[:] = []
    val_ar_per_epoch_list[:] = []

def update_epochs(num_epochs):
    reset_arr(arr_epochs)
    for i in range (num_epochs):
        arr_epochs.append(i)

def iter_per_epoch(data_list, epochs):
    len_data = len(data_list)
    len_epochs = len(epochs)

    return len_data / len_epochs

def add_data_to_data_arr(data_arr, data):
    data_arr.append(data)

def add_data_arr_to_data_list(data_list, data_arr):
    data_list.append(data_arr.copy())
    
def write_output_to_file_txt(file_name, save_dir_folder, num_epochs, nums_format, full_metrics):
    with open(f'{save_dir_folder}/{file_name}.txt', 'w') as f:
        space_separator = ' '
        metrics_data_per_epochs = []

        for i in range(num_epochs):
            reset_arr(metrics_data_per_epochs)
            if full_metrics == True:
                len_full_train_loss = len(train_loss_per_epoch)
                len_full_val_loss = len(val_loss_per_epoch)
                
                for j in range(len_full_train_loss):
                    add_data_to_data_arr(metrics_data_per_epochs, '{:.{}f}'.format(train_loss_per_epoch[j], nums_format))

                for j in range(len_full_val_loss):
                    if j == 0:
                        add_data_to_data_arr(metrics_data_per_epochs, '\n' + '{:.{}f}'.format(val_loss_per_epoch[j], nums_format))
                    else:
                        add_data_to_data_arr(metrics_data_per_epochs, '{:.{}f}'.format(val_loss_per_epoch[j], nums_format)) 
            else:
                add_data_to_data_arr(metrics_data_per_epochs, '{:.{}f}'.format(mean_train_loss_per_epoch[i], nums_format))
                add_data_to_data_arr(metrics_data_per_epochs, '{:.{}f}'.format(mean_val_loss_per_epoch[i], nums_format))
            
            if full_metrics == True:
                add_data_to_data_arr(metrics_data_per_epochs, '\n' + '{:.{}f}'.format(val_ap50_per_epoch[i], nums_format))
            else:
                add_data_to_data_arr(metrics_data_per_epochs, '{:.{}f}'.format(val_ap50_per_epoch[i], nums_format))
            add_data_to_data_arr(metrics_data_per_epochs, '{:.{}f}'.format(val_ap75_per_epoch[i], nums_format))
            add_data_to_data_arr(metrics_data_per_epochs, '{:.{}f}'.format(val_ap_per_epoch[i], nums_format))
            add_data_to_data_arr(metrics_data_per_epochs, '{:.{}f}'.format(val_ar_per_epoch[i], nums_format))

            line_data = space_separator.join(metrics_data_per_epochs)
            f.write("%s\n" % line_data)  

def create_arr(num_max_in_arr):
    new_arr = []
    for i in range (num_max_in_arr):
        add_data_to_data_arr(new_arr, i)
    return new_arr

def scale_arr(input_arr, ratio_scale):
    len_input_arr = len(input_arr)
    for i in range(len_input_arr):
        input_arr[i] = (input_arr[i] / ratio_scale)

def minor_tick_y(y, pos):
    if not y % 1.0:
        return ""
    return "%.2f" % y

def plot_data_per_epoch_per_hyper_params(arr_data, y_label, plt_name, plt_title, hyper_list, y_zoom_size, TRAIN_TYPE):

    arr_data_buffer = arr_data.copy()
    global HYPERPARAMETER_CHANGE

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    ax.xaxis.set_major_locator(MultipleLocator(1.000))
    # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(1.000))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    # ax.yaxis.set_minor_formatter(FuncFormatter(minor_tick_y))

    ax.tick_params(which='major', width=1.0)
    ax.tick_params(which='major', length=10)
    ax.tick_params(which='minor', width=1.0, labelsize=10)
    ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

    #max_data = 0
    i = -1

    for data in arr_data_buffer:
        i += 1
        len_data = len(data)
        num_iter = iter_per_epoch(data, arr_epochs)

        x_array = create_arr(len_data)
        scale_arr(x_array, num_iter)

        print(HYPERPARAMETER_CHANGE)
        print(hyper_list)
        print(HYPER_PARAMS_NAME_SHORTCUT[HYPERPARAMETER_CHANGE])
        if len(hyper_list) == 1 and 'val_accuracy' == hyper_list[0]:
            default_values = -1
        else:
            default_values = float(hyper_list[i].replace(f'{HYPER_PARAMS_NAME_SHORTCUT[HYPERPARAMETER_CHANGE]}=', ''))
        print(data)
        for j in range(len_data):
            data[j] *= y_zoom_size
        print(data)
        if default_values == base_hyperparameters[HYPERPARAMETER_CHANGE]:   
            ax.plot(x_array, data, c='#FF0000', lw=2, label=hyper_list[i])
        else:  
            ax.plot(x_array, data, lw=1, label=hyper_list[i]) 

    # Set font
    font = FontProperties()
    font.set_name('DejaVu Sans')
    font.set_size(20)
    font.setweight='bold'

    # Set title, xlabel, ylabel
    ax.set_title(plt_title, fontproperties=font, verticalalignment='bottom')  
    ax.set_xlabel('epochs')
    ax.set_ylabel(y_label)

    # Set legend
    ax.legend(ncol=5)
    
    # Save figure
    plt.savefig(f'../visualize/{TRAIN_TYPE}/{get_order_train_eval()}/{plt_name}.jpg', dpi=600)

def plot_train_and_val_data_per_epoch(val_data, y_label, plt_name, plt_title, label_train, label_val, TRAIN_TYPE, train_data=None):
    
    val_data_buffer = val_data.copy()
    if not train_data == None:
        train_data_buffer = train_data.copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))
    #ax = fig.add_subplot(1, 1, 1, aspect=1)

    ax.xaxis.set_major_locator(MultipleLocator(1.000))
    #ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(1.000))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_formatter(FuncFormatter(minor_tick_y))

    ax.tick_params(which='major', width=1.0)
    ax.tick_params(which='major', length=10)
    ax.tick_params(which='minor', width=1.0, labelsize=10)
    ax.tick_params(which='minor', length=5, labelsize=10, labelcolor='0.25')

    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

    if not train_data == None:
        max_data_train = max(train_data_buffer)
    else:
        max_data_train = -1
    max_data_val = max(val_data_buffer)

    max_data = max_data_train if max_data_train > max_data_val else max_data_val

    if not train_data == None:
        len_train_data = len(train_data_buffer)
        num_iter_train = iter_per_epoch(train_data_buffer, arr_epochs)
        x_train_array = create_arr(len_train_data)
        scale_arr(x_train_array, num_iter_train)
    
    len_val_data = len(val_data_buffer)
    num_iter_val = iter_per_epoch(val_data_buffer, arr_epochs)
    x_val_array = create_arr(len_val_data)
    scale_arr(x_val_array, num_iter_val)

    if not train_data == None:
        if num_iter_train > 1:
            x_train_array.append(len(arr_epochs))
            train_data_buffer.append(train_data_buffer[-1])
    if num_iter_val > 1:
        x_val_array.append(len(arr_epochs))
        val_data_buffer.append(val_data_buffer[-1])

    if not train_data == None:
        ax.plot(x_train_array, train_data_buffer, c=(1.00, 0.25, 0.25), lw=1, label=label_train, scaley=True)
    ax.plot(x_val_array, val_data_buffer, c=(0.25, 0.25, 1.00), lw=1, label=label_val, scaley=True)

    #ax.set_xlim(-1, len(arr_epochs) + 1)
    #ax.set_ylim(0, round(max_data) + 1)
    
    # Set font
    font = FontProperties()
    font.set_name('DejaVu Sans')
    font.set_size(20)
    font.setweight='bold'

    # Set title, xlabel, ylabel
    ax.set_title(plt_title, fontproperties=font, verticalalignment='bottom')  
    ax.set_xlabel('epochs')
    ax.set_ylabel(y_label)

    # Set legend
    ax.legend(ncol=2)

    # Save figure
    plt.savefig(f'../visualize/{TRAIN_TYPE}/{get_order_train_eval()}/{plt_name}.jpg', dpi=600)
