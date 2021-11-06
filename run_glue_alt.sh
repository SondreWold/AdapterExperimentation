export TASK_NAME=mnli
MODEL="output/test_1/"
python -u ex/run_glue_alt.py \
  --model_name_or_path $MODEL \
  --tokenizer_name bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --train_adapter \
  --save_steps 10000 \
  --adapter_config houlsby \
  --output_dir tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --tune_both \
