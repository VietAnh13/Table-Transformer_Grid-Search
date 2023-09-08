export FOLDER_SETUP='../setup'
export SETUP_FILE_SH='setup.sh'
export SETUP_FILE_SH_DIR="$FOLDER_SETUP/$SETUP_FILE_SH"
export SETUP_FILE_TXT='setup.txt'
export SETUP_FILE_TXT_DIR="$FOLDER_SETUP/$SETUP_FILE_TXT"

chmod 777 $FOLDER_SETUP
chmod +x $SETUP_FILE_SH_DIR
$SETUP_FILE_SH_DIR

export MODE_RUN=$(head -n 1 $SETUP_FILE_TXT_DIR)
if [ $MODE_RUN = 0 ]; then export MODE_RUN_NAME='default mode'
elif [ $MODE_RUN = 1 ]; then export MODE_RUN_NAME='visualize mode'
elif [ $MODE_RUN = 2 ]; then export MODE_RUN_NAME='grid search mode'; fi
export ORDER=$(sed -n '4p' $SETUP_FILE_TXT_DIR)
export SHORT_MODE=$(sed -n '2p' $SETUP_FILE_TXT_DIR)
if [ $SHORT_MODE = 'T' ]; then export MODE='train'
elif [ $SHORT_MODE = 'E' ]; then export MODE='eval'; fi
export NOHUP_OUT_FOLDER='../nohup_out'
export TYPE_NOHUP_OUT=$(sed -n '3p' $SETUP_FILE_TXT_DIR)
export NOHUP_OUT_DIR="$NOHUP_OUT_FOLDER/$TYPE_NOHUP_OUT/$MODE"
export NOHUP_OUT_FILE="nohup_${MODE}_${ORDER}.out"

echo "mode_run: $MODE_RUN_NAME"
echo "default_mode: $MODE"
echo "nohup_out_dir: $NOHUP_OUT_DIR"
echo "nohup_out_file: $NOHUP_OUT_FILE"
echo ""

chmod 777 $NOHUP_OUT_DIR
chmod +x $FOLDER_SETUP/$MODE.sh
nohup $FOLDER_SETUP/$MODE.sh >> $NOHUP_OUT_DIR/$NOHUP_OUT_FILE 2>&1 &