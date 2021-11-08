export TASK_NAME=ax
MODEL="tmp/mnli/"
python -u ex/run_glue_alt.py \
  --model_name_or_path $MODEL \
  --tokenizer_name bert-base-uncased \
  --task_name $TASK_NAME \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir tmp/$TASK_NAME/ \
  --overwrite_output_dir \
