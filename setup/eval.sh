export SETUP_FILE_TXT_DIR='../setup/setup.txt'

export ORDER=$(sed -n '5p' $SETUP_FILE_TXT_DIR)
export TABLE_TYPE=$(sed -n '7p' $SETUP_FILE_TXT_DIR)
export TRAIN_TYPE=$(sed -n '6p' $SETUP_FILE_TXT_DIR)

export MODE='eval'
if [ $TRAIN_TYPE = 'TD' ]; then export DATA_TYPE='detection' 
elif [ $TRAIN_TYPE = 'TSR' ]; then export DATA_TYPE='structure'; fi
export CONFIG_FILE="${DATA_TYPE}_config.json"
export DATA_ROOT_DIR="../data/Table_Type_$TABLE_TYPE/$TRAIN_TYPE"
if [ $TRAIN_TYPE = 'TSR' ]; then export TABLE_WORDS_DIR="$DATA_ROOT_DIR/words";
elif [ $TRAIN_TYPE = 'TD' ]; then export TABLE_WORDS_DIR=''; fi
if [ $TRAIN_TYPE = 'TSR' ]; then export __TABLE_WORDS_DIR="--table_words_dir $TABLE_WORDS_DIR";
elif [ $TRAIN_TYPE = 'TD' ]; then export __TABLE_WORDS_DIR=''; fi
export LOWER_TRAIN_TYPE="$(echo $TRAIN_TYPE | tr '[:upper:]' '[:lower:]')"

export MODE_RUN=$(head -n 1 $SETUP_FILE_TXT_DIR)
if [ $MODE_RUN = 0 ]; then export MODE_RUN_NAME='default_mode'
elif [ $MODE_RUN = 1 ]; then export MODE_RUN_NAME='visualize_mode'
elif [ $MODE_RUN = 2 ]; then export MODE_RUN_NAME='grid_search_mode'; fi
export MODEL_SAVE_DIR="$DATA_ROOT_DIR/output/$MODE_RUN_NAME/model_${LOWER_TRAIN_TYPE}_${ORDER}"
export MODEL_LOAD_PATH_FILE_TXT_DIR="$MODEL_SAVE_DIR/weights_save_dir.txt"
export MODEL_LOAD_PATH=$(head -n 1 $MODEL_LOAD_PATH_FILE_TXT_DIR)

export DEBUG_SAVE_DIR="../debug_eval/debug_${LOWER_TRAIN_TYPE}_tt${TABLE_TYPE}_${ORDER}"
export DEVICE='cpu'
export BATCH_SIZE=1

echo "order: $ORDER"
echo "table_type: $TABLE_TYPE"
echo "mode: $MODE"
echo "train_type: $TRAIN_TYPE"
echo "data_type: $DATA_TYPE"
echo "config_file: $CONFIG_FILE"
echo "data_root_dir: $DATA_ROOT_DIR"
if [ $TRAIN_TYPE = 'TSR' ]; then echo "table_words_dir: $TABLE_WORDS_DIR"; fi
echo "model_load_path: $MODEL_LOAD_PATH"
echo "debug_save_dir: $DEBUG_SAVE_DIR"
echo "device: $DEVICE"
echo "batch_size: $BATCH_SIZE"

python main.py --mode $MODE --data_type $DATA_TYPE --config_file $CONFIG_FILE --data_root_dir $DATA_ROOT_DIR --model_load_path $MODEL_LOAD_PATH --debug --debug_save_dir $DEBUG_SAVE_DIR --device $DEVICE --batch_size $BATCH_SIZE $__TABLE_WORDS_DIR
# python main.py --mode $MODE --data_type $DATA_TYPE --config_file $CONFIG_FILE --data_root_dir $DATA_ROOT_DIR --model_load_path $MODEL_LOAD_PATH --debug_save_dir $DEBUG_SAVE_DIR --device $DEVICE --batch_size $BATCH_SIZE $__TABLE_WORDS_DIR