export SETUP_FILE_TXT_DIR='../setup/setup.txt'

export ORDER=$(sed -n '5p' $SETUP_FILE_TXT_DIR)
export TABLE_TYPE=$(sed -n '7p' $SETUP_FILE_TXT_DIR)
export TRAIN_TYPE=$(sed -n '6p' $SETUP_FILE_TXT_DIR)
if [ $TRAIN_TYPE = 'TD' ]; then export DATA_TYPE='detection' 
elif [ $TRAIN_TYPE = 'TSR' ]; then export DATA_TYPE='structure'; fi
export CONFIG_FILE="${DATA_TYPE}_config.json"
export DATA_ROOT_DIR="../data/Table_Type_$TABLE_TYPE/$TRAIN_TYPE"
export MODEL_LOAD_PATH="../weights/pubtables1m_${DATA_TYPE}_detr_r18.pth"
export MODE_RUN=$(head -n 1 $SETUP_FILE_TXT_DIR)
if [ $MODE_RUN = 0 ]; then export MODE_RUN_NAME='default_mode'
elif [ $MODE_RUN = 1 ]; then export MODE_RUN_NAME='visualize_mode'
elif [ $MODE_RUN = 2 ]; then export MODE_RUN_NAME='grid_search_mode'; fi
export LOWER_TRAIN_TYPE="$(echo $TRAIN_TYPE | tr '[:upper:]' '[:lower:]')"
export MODEL_SAVE_DIR="$DATA_ROOT_DIR/output/$MODE_RUN_NAME/model_${LOWER_TRAIN_TYPE}_${ORDER}"

echo "order: $ORDER"
echo "table_type: $TABLE_TYPE"
echo "train_type: $TRAIN_TYPE"
echo "data_type: $DATA_TYPE"
echo "config_file: $CONFIG_FILE"
echo "data_root_dir: $DATA_ROOT_DIR"
echo "model_load_path: $MODEL_LOAD_PATH"
echo "model_save_dir: $MODEL_SAVE_DIR"

#python main.py --data_type $DATA_TYPE --config_file $CONFIG_FILE --data_root_dir $DATA_ROOT_DIR --model_load_path $MODEL_LOAD_PATH --model_save_dir $MODEL_SAVE_DIR
python main.py --data_type $DATA_TYPE --config_file $CONFIG_FILE --data_root_dir $DATA_ROOT_DIR --model_load_path $MODEL_LOAD_PATH --load_weights_only --model_save_dir $MODEL_SAVE_DIR
#python main.py --data_type $DATA_TYPE --config_file $CONFIG_FILE --data_root_dir $DATA_ROOT_DIR --model_load_path $MODEL_LOAD_PATH --model_save_dir $MODEL_SAVE_DIR