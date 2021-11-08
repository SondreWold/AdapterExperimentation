import argparse
import logging
import math
import os
import random
import sys
import numpy as np
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    Trainer,
    AdapterTrainer,
    AutoConfig,
    EvalPrediction,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a transformers model on a diagnostic classification task from GLUE")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="A seed for reproducible training.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    task_name = "ax"
    raw_dataset = load_dataset("glue", task_name)

    label_list = raw_dataset["test"].features["label"].names
    print(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}
    set_seed(args.seed)

    accelerator = Accelerator()

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        finetuning_task="ax"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    model.train_adapter(["mlm"])  # activate adapter
    model.set_active_adapters(["mlm"])

    sentence1_key = "premise"
    sentence2_key = "hypothesis"

    padding = "max_length"

    metric = load_metric("matthews_correlation")

    def compute_metrics(p: EvalPrediction):
        is_regression = False
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        preds = np.squeeze(
            preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(
                list(result.values())).item()
        return result

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding,
                           max_length=128, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1)
                               for l in examples["label"]]
        return result

    processed_datasets = raw_dataset.map(
        preprocess_function, batched=True, remove_columns=raw_dataset["test"].column_names
    )

    predict_dataset = processed_datasets["test"]
    data_collator = default_data_collator

    # Log a few random samples from the test set:
    for index in random.sample(range(len(predict_dataset)), 3):
        logger.info(
            f"Sample {index} of the training set: {predict_dataset[index]}.")

    logger.info("*** Predict ***")

    pred_dataloader = DataLoader(
        predict_dataset, collate_fn=data_collator, batch_size=8
    )

    trainer_class = AdapterTrainer
    trainer = trainer_class(
        model=model,
        train_dataset=None,
        eval_dataset=predict_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    model.eval()

    predict_dataset = predict_dataset.remove_columns("label")
    predictions = trainer.predict(
        predict_dataset, metric_key_prefix="predict").predictions
    predictions = np.argmax(predictions, axis=1)

    output_predict_file = os.path.join(
        args.output_dir, f"predict_results_{task_name}.txt")
    if trainer.is_world_process_zero():
        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Predict results {task_name} *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                item = label_list[item]
                writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
