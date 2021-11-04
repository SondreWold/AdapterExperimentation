export TASK_NAME=mrpc
MODEL="output/test_1/"
python -u ex/run_glue_alt.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 10.0 \
  --output_dir tmp/$TASK_NAME/ \
  --overwrite_output_dir \
