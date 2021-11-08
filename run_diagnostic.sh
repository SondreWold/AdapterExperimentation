export TASK_NAME=ax
MODEL="models/mnli/"
python -u ex/run_diagnostic.py \
  --model_name_or_path $MODEL \
  --tokenizer_name bert-base-uncased \
  --output_dir results/diagnostic/ \
