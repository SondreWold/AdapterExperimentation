export TASK_NAME=mnli
MODEL="output/test_1/"
python -u ex/run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name bert-base-uncased \
  --task_name $TASK_NAME \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir tmp/$TASK_NAME/ \
  --tune_both true \
