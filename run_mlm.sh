TRAIN_FILE="./data/cn_train.txt"
VAL_FILE="./data/cn_validation.txt"
MODEL_TYPE="bert-base-uncased"
TRAINING_FOLDER="./training"

python $TRAINING_FOLDER/mlm.py \
    --model_name_or_path $MODEL_TYPE \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --line_by_line true \
    --output_dir /tmp/test-mlm