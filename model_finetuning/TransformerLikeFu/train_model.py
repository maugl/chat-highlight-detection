import os
import time
from collections import ChainMap

import datasets
import numpy as np
import torch
from transformers import RobertaModel, PreTrainedModel, EarlyStoppingCallback, IntervalStrategy
import json

from transformers import RobertaTokenizerFast
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from hub_token import HUB_TOKEN


def train_model(dataset, tokenizer_path, lm_path, run_id, output_dir):
    tok = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(lm_path, num_labels=1)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_steps=20000,
        save_total_limit=4,
        evaluation_strategy=IntervalStrategy("steps"),
        eval_steps=20000,
        logging_steps=500,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        label_names=["labels"],

        push_to_hub=True,
        hub_model_id=f"{lm_path}-finetuned-highlight-detection",  # optional, will default to the name of your output directory
        hub_token=HUB_TOKEN
    )

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tok,

        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data
    try:
        with open(f"/{training_args.logging_dir}/log_history_{run_id}.json", "w") as out_file:
            json.dump(trainer.state.log_history, out_file, indent=4) # might have to change this to copy back to storage
    except Exception as e:
        print("cannot create log_history")
        print(e)

    # https://github.com/philschmid/huggingface-sagemaker-workshop-series/blob/345374941c55aa32f95d5993f7a7fc461e18f907/workshop_4_distillation_and_acceleration/scripts/train.py#L133
    # some issue with asynchronous pushes at the same time
    time.sleep(180)
    trainer.push_to_hub()
    time.sleep(180)
    trainer.save_model()

    final_eval = trainer.evaluate()
    try:
        with open(f"{training_args.logging_dir}/final_eval.txt", "w") as out_file:
            out_file.write(str(final_eval))
    except Exception as e:
        print("cannot compute final eval")
        print(e)


# === eval metrics ===
def define_metrics():
    f1_metric = datasets.load_metric("f1")
    p_metric = datasets.load_metric("precision")
    r_metric = datasets.load_metric("recall")
    metrics = [f1_metric, p_metric, r_metric]
    return metrics


def compute_metrics(p):
    # https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    refs = p.label_ids

    preds = torch.Tensor(np.ravel(preds))
    refs = torch.tensor(np.ravel(refs))

    metrics = define_metrics()
    results = dict(ChainMap(*[m.compute(predictions=preds, references=refs) for m in metrics]))
    return results


def preprocess_logits_for_metrics(logits, labels):
    return torch.round(torch.sigmoid(logits))