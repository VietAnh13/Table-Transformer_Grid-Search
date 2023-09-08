"""
Copyright (C) 2021 Microsoft Corporation
"""
import os
import argparse
import json
from datetime import datetime
import string
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# CODE ADD FROM HERE
import shutil
sys.path.extend(["../setup", "../detr"])
from config import *
# from visualize import *
# from grid_search import *
from run_mode import *
# TO HERE

# sys.path.append("../detr")
from engine import evaluate, train_one_epoch
from models import build_model
import util.misc as utils
import datasets.transforms as R

import table_datasets as TD
from table_datasets import PDFTablesDataset
from eval import eval_coco


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_dir',
                        required=True,
                        help="Root data directory for images and labels")
    parser.add_argument('--config_file',
                        required=True,
                        help="Filepath to the config containing the args")
    parser.add_argument('--backbone',
                        # default='resnet18',
                        help="Backbone for the model", type=str)
    parser.add_argument(
        '--data_type',
        choices=['detection', 'structure'],
        default='structure',
        help="toggle between structure recognition and table detection")
    parser.add_argument('--model_load_path', help="The path to trained model")
    parser.add_argument('--load_weights_only', action='store_true')
    parser.add_argument('--model_save_dir', help="The output directory for saving model params and checkpoints")
    parser.add_argument('--metrics_save_filepath',
                        help='Filepath to save grits outputs',
                        default='')
    parser.add_argument('--debug_save_dir',
                        help='Filepath to save visualizations',
                        default='debug')                        
    parser.add_argument('--table_words_dir',
                        help="Folder containg the bboxes of table words")
    parser.add_argument('--mode',
                        choices=['train', 'eval'],
                        default='train',
                        help="Modes: training (train) and evaluation (eval)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_drop', type=int)
    parser.add_argument('--lr_gamma', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--checkpoint_freq', default=1, type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--train_max_size', type=int)
    parser.add_argument('--val_max_size', type=int)
    parser.add_argument('--test_max_size', type=int)
    parser.add_argument('--eval_pool_size', type=int, default=1)
    parser.add_argument('--eval_step', type=int, default=1)

    return parser.parse_args()


def get_transform(data_type, image_set):
    if data_type == 'structure':
        return TD.get_structure_transform(image_set)
    else:
        return TD.get_detection_transform(image_set)


def get_class_map(data_type):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6
        }
    else:
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map


def get_data(args):
    """
    Based on the args, retrieves the necessary data to perform training, 
    evaluation or GriTS metric evaluation
    """
    # Datasets
    print("loading data")
    class_map = get_class_map(args.data_type)

    if args.mode == "train":
        dataset_train = PDFTablesDataset(
            os.path.join(args.data_root_dir, "train"),
            get_transform(args.data_type, "train"),
            do_crop=False,
            max_size=args.train_max_size,
            include_eval=False,
            max_neg=0,
            make_coco=False,
            image_extension=".jpg",
            xml_fileset="train_filelist.txt",
            class_map=class_map)
        dataset_val = PDFTablesDataset(os.path.join(args.data_root_dir, "val"),
                                       get_transform(args.data_type, "val"),
                                       do_crop=False,
                                       max_size=args.val_max_size,
                                       include_eval=False,
                                       make_coco=True,
                                       image_extension=".jpg",
                                       xml_fileset="val_filelist.txt",
                                       class_map=class_map)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                            args.batch_size,
                                                            drop_last=True)

        data_loader_train = DataLoader(dataset_train,
                                       batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn,
                                       num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val,
                                     2 * args.batch_size,
                                     sampler=sampler_val,
                                     drop_last=False,
                                     collate_fn=utils.collate_fn,
                                     num_workers=args.num_workers)
        return data_loader_train, dataset_train, data_loader_val, dataset_val, len(
            dataset_train)

    elif args.mode == "eval":

        dataset_test = PDFTablesDataset(os.path.join(args.data_root_dir,
                                                     "test"),
                                        get_transform(args.data_type, "val"),
                                        do_crop=False,
                                        max_size=args.test_max_size,
                                        make_coco=True,
                                        include_eval=True,
                                        image_extension=".jpg",
                                        xml_fileset="test_filelist.txt",
                                        class_map=class_map)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = DataLoader(dataset_test,
                                      2 * args.batch_size,
                                      sampler=sampler_test,
                                      drop_last=False,
                                      collate_fn=utils.collate_fn,
                                      num_workers=args.num_workers)
        return data_loader_test, dataset_test

    elif args.mode == "grits" or args.mode == "grits-all":
        dataset_test = PDFTablesDataset(os.path.join(args.data_root_dir,
                                                     "test"),
                                        RandomMaxResize(1000, 1000),
                                        include_original=True,
                                        max_size=args.max_test_size,
                                        make_coco=False,
                                        image_extension=".jpg",
                                        xml_fileset="test_filelist.txt",
                                        class_map=class_map)
        return dataset_test


def get_model(args, device):
    """
    Loads DETR model on to the device specified.
    If a load path is specified, the state dict is updated accordingly.
    """
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.model_load_path:
        print("loading model from checkpoint")
        loaded_state_dict = torch.load(args.model_load_path,
                                       map_location=device)
        model_state_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in loaded_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict, strict=True)
    return model, criterion, postprocessors


def train(args, model, criterion, postprocessors, device):
    """
    Training loop
    """

    print("loading data")
    dataloading_time = datetime.now()
    data_loader_train, dataset_train, data_loader_val, dataset_val, train_len = get_data(args)
    print("finished loading data in :", datetime.now() - dataloading_time)

    model_without_ddp = model
    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr":
            args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.lr_drop,
                                                   gamma=args.lr_gamma)

    max_batches_per_epoch = int(train_len / args.batch_size)
    print("Max batches per epoch: {}".format(max_batches_per_epoch))

    resume_checkpoint = False
    if args.model_load_path:
        checkpoint = torch.load(args.model_load_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)

        if not args.load_weights_only and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resume_checkpoint = True
        elif args.load_weights_only:
            print("*** WARNING: Resuming training and ignoring optimzer state. "
                  "Training will resume with new initialized values. "
                  "To use current optimizer state, remove the --load_weights_only flag.")
        else:
            print("*** ERROR: Optimizer state of saved checkpoint not found. "
                  "To resume training with new initialized values add the --load_weights_only flag.")
            raise Exception("ERROR: Optimizer state of saved checkpoint not found. Must add --load_weights_only flag to resume training without.")          
        
        if not args.load_weights_only and 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
        elif args.load_weights_only:
            print("*** WARNING: Resuming training and ignoring previously saved epoch. "
                  "To resume from previously saved epoch, remove the --load_weights_only flag.")
        else:
            print("*** WARNING: Epoch of saved model not found. Starting at epoch {}.".format(args.start_epoch))

    # Use user-specified save directory, if specified
    if args.model_save_dir:
        output_directory = args.model_save_dir
    # If resuming from a checkpoint with optimizer state, save into same directory
    elif args.model_load_path and resume_checkpoint:
        output_directory = os.path.split(args.model_load_path)[0]
    # Create new save directory
    else:
        run_date = datetime.now().strftime("%Y%m%d%H%M%S")
        output_directory = os.path.join(args.data_root_dir, "output", run_date)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Output directory: ", output_directory)
    model_save_path = os.path.join(output_directory, 'model.pth')
    print("Output model path: ", model_save_path)
    if not resume_checkpoint and os.path.exists(model_save_path):
        print("*** WARNING: Output model path exists but is not being used to resume training; training will overwrite it.")

    if args.start_epoch >= args.epochs:
        print("*** WARNING: Starting epoch ({}) is greater or equal to the number of training epochs ({}).".format(
            args.start_epoch, args.epochs
        ))

    print("Start training")
    start_time = datetime.now()

    # CODE ADD FROM HERE
    NUM_FREEZED_PARAMS = get_num_freezed_params()
    # NOTE: Checking freezing RIGHT BEFORE TRAINING.
    log('Checking freezing RIGHT BEFORE TRAINING!!!')
    check_freezing(NUM_FREEZED_PARAMS, model.backbone)
    count_num_freezed_params(model,'ALL') 
    count_num_freezed_params(model.backbone,'backbone') 
    # TO HERE

    for epoch in range(args.start_epoch, args.epochs):
        print('-' * 100)

        epoch_timing = datetime.now()
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            max_batches_per_epoch=max_batches_per_epoch,
            print_freq=1000)
        print("Epoch completed in ", datetime.now() - epoch_timing)

        lr_scheduler.step()

        pubmed_stats, coco_evaluator = evaluate(model, criterion,
                                                postprocessors,
                                                data_loader_val, dataset_val,
                                                device, None)
        print("pubmed: AP50: {:.3f}, AP75: {:.3f}, AP: {:.3f}, AR: {:.3f}".
              format(pubmed_stats['coco_eval_bbox'][1],
                     pubmed_stats['coco_eval_bbox'][2],
                     pubmed_stats['coco_eval_bbox'][0],
                     pubmed_stats['coco_eval_bbox'][8]))

        # CODE ADD FROM HERE
        current_index = get_current_order() + get_standard_deviation_order()
        
        if RUN_MODE_STATUS[1]:
            add_data_to_data_arr(mean_train_loss_per_epoch, np.mean(train_loss_per_epoch))
            add_data_to_data_arr(mean_val_loss_per_epoch, np.mean(val_loss_per_epoch))
            add_data_to_data_arr(val_ap50_per_epoch, pubmed_stats['coco_eval_bbox'][1])
            add_data_to_data_arr(val_ap75_per_epoch, pubmed_stats['coco_eval_bbox'][2])
            add_data_to_data_arr(val_ap_per_epoch, pubmed_stats['coco_eval_bbox'][0])
            add_data_to_data_arr(val_ar_per_epoch, pubmed_stats['coco_eval_bbox'][8])   
        
        VAL_AP50 = pubmed_stats['coco_eval_bbox'][1]
        VAL_AP75 = pubmed_stats['coco_eval_bbox'][2]
        VAL_AP = pubmed_stats['coco_eval_bbox'][0]
        VAL_AR = pubmed_stats['coco_eval_bbox'][8]

        metrics_values = create_metrics_values(VAL_AP50, VAL_AP75, VAL_AP, VAL_AR)
        update_highest_metrics_epochs_val_accuracy(metrics_values, epoch + 1)
        TRAIN_TYPE = update_train_type()
        ORDER_TRAIN_EVAL = get_order_train_eval()

        # Save data metrics val accuracy hyper params (MVAHP) to file txt
        file_name_MVAHP = f'metrics_val_accuracy_hyper_params_{current_index + 1}'
        save_dir_MVAHP = f'../run_mode_log/{get_run_mode_name()[get_run_mode()]}/{TRAIN_TYPE}/{ORDER_TRAIN_EVAL}'
        first_line_data_hyper_params_value = create_line_hyper_params_to_write(get_current_hyper_params_value())

        if epoch == 0:
            os.makedirs(save_dir_MVAHP, exist_ok=True)
            write_data_to_file_txt(file_name_MVAHP, save_dir_MVAHP, first_line_data_hyper_params_value, True)
        
        line_data_metrics_values = create_line_data_to_write(metrics_values)
        write_data_to_file_txt(file_name_MVAHP, save_dir_MVAHP, line_data_metrics_values, False)

        # Metrics values ​​selected at epoch have the highest AP75 value on the val dataset
        # Followed by AP, AP50 and AR in order if AP75 is equal
        if epoch == args.epochs - 1:
            # Save highest metrics values ​​epochs list (HMVEL) on the val dataset to file txt
            file_name_HMVEL = f'highest_metrics_values_epochs_list'
            save_dir_HMVEL = f'../run_mode_log/{get_run_mode_name()[get_run_mode()]}/{TRAIN_TYPE}/{ORDER_TRAIN_EVAL}'
            os.makedirs(save_dir_HMVEL, exist_ok=True)
            line_data_HMVEL = create_line_data_to_write(get_highest_metrics_epochs_val_accuracy()) + ' ' + first_line_data_hyper_params_value
            
            if current_index == 0:
                write_data_to_file_txt(file_name_HMVEL, save_dir_HMVEL, line_data_HMVEL, True)
            else:
                write_data_to_file_txt(file_name_HMVEL, save_dir_HMVEL, line_data_HMVEL, False)
            
            update_max_metrics_values(get_highest_metrics_epochs_val_accuracy(), get_max_metrics_in_val(), current_index)

            # Save highest data metrics current (HDMC) to file txt
            file_name_HDMC = f'highest_metrics_val_accuracy_current'
            save_dir_HDMC = f'../run_mode_log/{get_run_mode_name()[get_run_mode()]}/{TRAIN_TYPE}/{ORDER_TRAIN_EVAL}'
            os.makedirs(save_dir_HDMC, exist_ok=True)
            best_line_data_hyper_params_value = create_line_hyper_params_to_write(get_best_hyper_params_match())
            line_data_HDMC = create_line_data_to_write(get_max_metrics_in_val()) + ' ' + best_line_data_hyper_params_value
            write_data_to_file_txt(file_name_HDMC, save_dir_HDMC, line_data_HDMC, True)
        
        log_run_mode_state = f'False\n{current_index}\nFalse\n' + model_save_path
        write_state_run_mode(log_run_mode_state)
        # TO HERE

        # CODE FREEZING FROM HERE
        # Save current model training progress
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, model_save_path)
        # TO HERE                    

        # CODE FREEZING FROM HERE
        if RUN_MODE_STATUS[0] or RUN_MODE_STATUS[2]:
            # Save checkpoint for evaluation
            if (epoch + 1) % args.checkpoint_freq == 0:
                model_save_path_epoch = os.path.join(output_directory, 'model_' + str(epoch + 1) + '.pth')
                torch.save(model.state_dict(), model_save_path_epoch)
        # TO HERE 
    
    # CODE ADD FROM HERE
    # Keep the checkpoint with the highest AP75 score and delete the remaining checkpoints
    for epoch in range(0, args.epochs):
        weights_path_epoch = os.path.join(output_directory, 'model_' + str(epoch + 1) + '.pth')
        if epoch + 1 != get_epochs_highest_metrics():
           os.remove(weights_path_epoch)
        else:
            write_weights_save_dir(output_directory, weights_path_epoch)
            
    # Update success state for this current index
    write_state_run_mode(f'False\n{current_index}\nTrue\n')
    
    # Delete current model training progress
    os.remove(model_save_path)
    # TO HERE

    print('Total training time: ', datetime.now() - start_time)


def main():
    
    # CODE ADD FROM HERE TO FINETUNE (FT)
    start_time_run_mode = datetime.now()
    update_mode_run()
    TRAIN_TYPE = update_train_type()

    if RUN_MODE_STATUS[1]:
        reset_all_arr()
        hyper_list = []
        DATASET_NAME = 'Dataset BCTC Table Type 2 & 3'
        standard_deviation = 0
        range_size = 0
        flag_range_size = True
    
    for index in range(len_of_run_mode[RUN_MODE]):
        print('> [ Index:', index + get_standard_deviation_order(), '| Total:', len_of_run_mode[RUN_MODE] + get_standard_deviation_order(), '] <')
        update_current_order(index)
        update_progress_status()
        reset_highest_metrics_epochs_val_accuracy()

        cmd_args = get_args().__dict__
        config_args = json.load(open(cmd_args['config_file'], 'rb'))
        for key, value in cmd_args.items():
            if not key in config_args or not value is None:
                config_args[key] = value
        #config_args.update(cmd_args)
        args = type('Args', (object,), config_args)

        # CODE ADD FROM HERE
        finish_epochs_progress = get_finish_epochs_progress()

        if not finish_epochs_progress:
            args.model_load_path = get_last_model_save_dir()
            args.load_weights_only = False

        if RUN_MODE_STATUS[1]:
            HYPERPARAMETER_CHANGE = update_hyperparameter_change(index)
            
            range_size_state = 0
            for i in range(HYPERPARAMETER_RANGE_START, HYPERPARAMETER_CHANGE + 1, 1):
                range_size_state += len_hyperparameters[i]

            if flag_range_size == True:
                range_size = range_size_state
                flag_range_size = False
            else:
                if range_size != range_size_state:
                    standard_deviation = range_size
                    range_size = range_size_state
            
            relative_index = index - standard_deviation    

            HYPERPARAMETER_VALUE_LIST = []
            for j in range(len_base_hyperparameters):
                if base_hyperparameters_activate[j] == True:
                    HYPER_PARAMS_VALUE = hyperparameters_space[HYPERPARAMETER_CHANGE][relative_index]
                    HYPERPARAMETER_VALUE_LIST.append(HYPER_PARAMS_VALUE)
                else:
                    HYPERPARAMETER_VALUE_LIST.append(base_hyperparameters[j])

            hyper_legend = f'{HYPER_PARAMS_NAME_SHORTCUT[HYPERPARAMETER_CHANGE]}={HYPER_PARAMS_VALUE}'
            hyper_list.append(hyper_legend)

            args.lr = HYPERPARAMETER_VALUE_LIST[0]
            args.batch_size = HYPERPARAMETER_VALUE_LIST[1]
            args.epochs = HYPERPARAMETER_VALUE_LIST[2]
            NUM_FREEZED_PARAMS = update_num_freezed_params(HYPERPARAMETER_VALUE_LIST[3])

            update_epochs(args.epochs)
        
        if RUN_MODE_STATUS[2]:
            args.lr, args.batch_size, args.epochs, num_freezing_layer = get_hyper_params_in_search_space(index + get_standard_deviation_order())
            NUM_FREEZED_PARAMS = update_num_freezed_params(num_freezing_layer)
            if get_base_model_save_dir() == '':
                update_base_model_save_dir(args.model_save_dir)
            current_index = get_current_order() + get_standard_deviation_order()
            args.model_save_dir += f'/{current_index + 1}'
        
        if RUN_MODE_STATUS[0] or RUN_MODE_STATUS[1]:
            num_freezing_layer = get_num_freezed_params()
        hyper_params_arr = [args.lr, args.batch_size, args.epochs, num_freezing_layer]
        update_current_hyper_params_value(hyper_params_arr)

        # TO HERE

        print(args.__dict__)
        print('-' * 100)

        # Check for debug mode
        if args.mode == 'eval' and args.debug:
            print("Running evaluation/inference in DEBUG mode, processing will take longer. Saving output to: {}.".format(args.debug_save_dir))
            os.makedirs(args.debug_save_dir, exist_ok=True)

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("loading model")
        device = torch.device(args.device)
        model, criterion, postprocessors = get_model(args, device)

        # CODE ADD FROM HERE
        # ============================================
        # DEBUG: Check HOW MANY layers've already freezed.
        count_num_freezed_params(model,'ALL') # RES: 5/215
        count_num_freezed_params(model.backbone,'backbone') # RES:5/20

        """
        VERIFICATION:
        1. PyTorch docs about REQUIRES_GRAD: https://pytorch.org/docs/master/notes/autograd.html#setting-requires-grad
        2. TRAINING TIME AFTER FREEZING BACKBONE is FASTER than BEFORE.
        """

        # CONFIG: 2. Freezing SPECIFIED NUMBER OF LAYERS of BACKBONE.
        # TODO: DEFAULT, 5 FIRST LAYERS of BACKBONE is ALREADY FREEZED.
        if RUN_MODE_STATUS[0]:
            NUM_FREEZED_PARAMS = get_num_freezed_params()
        freeze_module(NUM_FREEZED_PARAMS, model.backbone)
        check_freezing(NUM_FREEZED_PARAMS, model.backbone)
        count_num_freezed_params(model,'ALL') 
        count_num_freezed_params(model.backbone,'backbone') 
        # TO HERE

        if args.mode == "train":
            train(args, model, criterion, postprocessors, device)
            if RUN_MODE_STATUS[1]:
                # CODE ADD FROM HERE
                # Write full metrics data per epochs to .txt file
                full_loss_file_name = f'{FIGURE_NAME_SHORTCUT[HYPERPARAMETER_CHANGE].lower()}_full_{TRAIN_TYPE.lower()}_{relative_index + 1}'
                full_loss_save_dir_folder = f'../run_mode_log/visualize_mode/{TRAIN_TYPE}/{ORDER_TRAIN_EVAL}'
                nums_full_format = 6
                write_output_to_file_txt(full_loss_file_name, full_loss_save_dir_folder, args.epochs, nums_full_format, True)

                # Write metrics data per epochs to .txt file
                txt_file_name = f'{FIGURE_NAME_SHORTCUT[HYPERPARAMETER_CHANGE].lower()}_simple_{TRAIN_TYPE.lower()}_{relative_index + 1}'
                txt_save_dir_folder = f'../run_mode_log/visualize_mode/{TRAIN_TYPE}/{ORDER_TRAIN_EVAL}'
                nums_simple_format = 3
                write_output_to_file_txt(txt_file_name, txt_save_dir_folder, args.epochs, nums_simple_format, False)

                # Update list metrics accuracy (AP75 val_dataset) & loss (mean val_loss)
                add_data_arr_to_data_list(val_ap75_per_epoch_list, val_ap75_per_epoch)
                add_data_arr_to_data_list(mean_val_loss_per_epoch_list, mean_val_loss_per_epoch)

                # Save figure full train_loss & val_loss (FTLVL)
                plt_name_FTLVL = f'Visualize_FTLVL_{TRAIN_TYPE}_TT23_GL_{FIGURE_NAME_SHORTCUT[HYPERPARAMETER_CHANGE]}_{relative_index + 1}'
                plt_title_FTLVL = f'TRANINING LOSS & VALIDATION LOSS FULL PER EPOCH ({hyper_legend})\n(Problems {TRAIN_TYPE} - {DATASET_NAME})'
                plot_train_and_val_data_per_epoch(val_loss_per_epoch, 'losses', plt_name_FTLVL, plt_title_FTLVL, 'train_loss', 'val_loss', TRAIN_TYPE, train_loss_per_epoch)

                # Save figure average train_loss & val_loss (ATLVL)
                plt_name_ATLVL = f'Visualize_ATLVL_{TRAIN_TYPE}_TT23_GL_{FIGURE_NAME_SHORTCUT[HYPERPARAMETER_CHANGE]}_{relative_index + 1}'
                plt_title_ATLVL = f'TRANINING LOSS & VALIDATION LOSS AVERAGE PER EPOCH ({hyper_legend})\n(Problems {TRAIN_TYPE} - {DATASET_NAME})'
                plot_train_and_val_data_per_epoch(mean_val_loss_per_epoch, 'losses', plt_name_ATLVL, plt_title_ATLVL, 'train_loss', 'val_loss', TRAIN_TYPE, mean_train_loss_per_epoch)

                # Save figure zoom accuracy (AP75 val_dataset) per epoch (ZAPE)
                plt_name_ZAPE = f'Visualize_ZAPE_{TRAIN_TYPE}_TT23_GL_{FIGURE_NAME_SHORTCUT[HYPERPARAMETER_CHANGE]}_{relative_index + 1}'
                plt_title_ZAPE = f'VALIDATION ACCURACY (AP75) PER EPOCH ({hyper_legend})\n(Problems {TRAIN_TYPE} - {DATASET_NAME})'
                data_list_ZAPE = []
                data_list_ZAPE.append(val_ap75_per_epoch)
                hyper_list_ZAPE = []
                hyper_list_ZAPE.append('val_accuracy')
                y_zoom_size_ZAPE = 100
                plot_data_per_epoch_per_hyper_params(data_list_ZAPE, 'val_acurracy', plt_name_ZAPE, plt_title_ZAPE, hyper_list_ZAPE, y_zoom_size_ZAPE, TRAIN_TYPE)
                
                # Reset arr metrics
                reset_arr(train_loss_per_epoch)
                reset_arr(val_loss_per_epoch)
                reset_arr(mean_train_loss_per_epoch)
                reset_arr(mean_val_loss_per_epoch)
                reset_arr(val_ap50_per_epoch)
                reset_arr(val_ap75_per_epoch)
                reset_arr(val_ap_per_epoch)
                reset_arr(val_ar_per_epoch)
                # TO HERE

                if index == range_size - 1:
                    # Save figure zoom accuracy (AP75 val_dataset) per epoch per hyper params (ZAPEPHP)
                    y_zoom_size_ZAPEPHP = 100
                    plt_name_ZAPEPHP = f'Visualize_ZAPEPHP_{TRAIN_TYPE}_TT23_GL_{FIGURE_NAME_SHORTCUT[HYPERPARAMETER_CHANGE]}_1'
                    plt_title_ZAPEPHP = f'ZOOM VALIDATION ACCURACY (AP75) PER EPOCH WITH DIFFERENT {HYPER_PARAMS_NAME_FULL[HYPERPARAMETER_CHANGE]}\n(Problems {TRAIN_TYPE} - {DATASET_NAME})'
                    plot_data_per_epoch_per_hyper_params(val_ap75_per_epoch_list, 'val_accuracy', plt_name_ZAPEPHP, plt_title_ZAPEPHP, hyper_list, y_zoom_size_ZAPEPHP, TRAIN_TYPE)
        
                    # Save figure zoom lost (mean val_loss) per epoch per hyper params (ZLPEPHP)
                    y_zoom_size_ZLPEPHP = 100
                    plt_name_ZLPEPHP = f'Visualize_ZLPEPHP_{TRAIN_TYPE}_TT23_GL_{FIGURE_NAME_SHORTCUT[HYPERPARAMETER_CHANGE]}_1'
                    plt_title_ZLPEPHP = f'ZOOM VALIDATION LOSS AVERAGE PER EPOCH WITH DIFFERENT {HYPER_PARAMS_NAME_FULL[HYPERPARAMETER_CHANGE]}\n(Problems {TRAIN_TYPE} - {DATASET_NAME})'
                    plot_data_per_epoch_per_hyper_params(mean_val_loss_per_epoch_list, 'val_losses', plt_name_ZLPEPHP, plt_title_ZLPEPHP, hyper_list, y_zoom_size_ZLPEPHP, TRAIN_TYPE)

                    # Reset arr metrics list
                    reset_arr(hyper_list)
                    reset_arr(val_ap75_per_epoch_list)
                    reset_arr(mean_val_loss_per_epoch_list)
                
        elif args.mode == "eval":
            data_loader_test, dataset_test = get_data(args)
            eval_coco(args, model, criterion, postprocessors, data_loader_test, dataset_test, device)
    
        # if index == 1:
        #     break

    cmd_args = get_args().__dict__
    config_args = json.load(open(cmd_args['config_file'], 'rb'))
    for key, value in cmd_args.items():
        if not key in config_args or not value is None:
            config_args[key] = value
    #config_args.update(cmd_args)
    args = type('Args', (object,), config_args)
    if get_base_model_save_dir() == '':
        update_base_model_save_dir(args.model_save_dir)
    
    # if RUN_MODE_STATUS[2]:
    #     TRAIN_TYPE = update_train_type()
    #     ORDER_TRAIN_EVAL = get_order_train_eval()
    #     file_name_HMVEL = f'highest_metrics_values_epochs_list'
    #     save_dir_HMVEL = f'../run_mode_log/{get_run_mode_name()[get_run_mode()]}/{TRAIN_TYPE}/{ORDER_TRAIN_EVAL}'
    #     epoch = 0
    #     with open(f'{save_dir_HMVEL}/{file_name_HMVEL}.txt') as f:
    #         lines = f.readlines()
    #         if len(lines) > 0:
    #             for line in lines:
    #                 epoch += 1
    #                 line_metrics = str(line).replace('\n', '')
    #                 current_line_metrics = line_metrics.split()
    #                 AP50_current = float(current_line_metrics[0])
    #                 AP75_current = float(current_line_metrics[1])
    #                 AP_current = float(current_line_metrics[2])
    #                 AR_current = float(current_line_metrics[3])
    #                 current_metrics = create_metrics_values(AP50_current, AP75_current, AP_current, AR_current)
    #                 if current_metrics == get_highest_metrics_epochs_val_accuracy():
    #                     break
                    
    #     for index in range(1, create_search_space(TRAIN_TYPE) + 1, 1):
    #         folder_path = f'{get_base_model_save_dir()}/{index}'
    #         if index == epoch:
    #             for file_name in os.listdir(folder_path):
    #                 if file_name.endswith('.pth'):
    #                     update_best_model_name(file_name)
    #                     break
    #             shutil.copy(f'{folder_path}/{get_best_model_name()}', get_base_model_save_dir())
    #             write_weights_save_dir(get_base_model_save_dir(), f'{get_base_model_save_dir()}/{get_best_model_name()}')
    #         shutil.rmtree(folder_path)

    write_state_run_mode('True\n')
    print('Total run mode time: ', datetime.now() - start_time_run_mode)

if __name__ == "__main__":
    main()
