TRAIN_FILE="./data/cn_train.txt"
VAL_FILE="./data/cn_validation.txt"
MODEL_TYPE="bert-base-uncased"
TRAINING_FOLDER="./training"
OUTPUT_DIR="./output/test_1/"

python $TRAINING_FOLDER/mlm.py \
    --model_name_or_path $MODEL_TYPE \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --output_dir $OUTPUT_DIR \
    --line_by_line true \
    --pad_to_max_length \
    --train_adapter true \
    --adapter_config "houlsby" \
    --max_train_steps 100000 \
    --output_dir /tmp/test-mlm
