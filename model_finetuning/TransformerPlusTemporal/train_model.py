import json
from collections import ChainMap
import time

import datasets
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, IntervalStrategy, EarlyStoppingCallback
from hub_token import HUB_TOKEN

from LanguageModelPlusTemporal import LanguageModelPlusTemporal


def train_model(dataset,
                lm_path,
                output_dir,
                additional_features_size,
                window_size,
                sequence_length,
                num_dist_categories,
                num_dist_steps,
                main_loss_ratio,
                run_id):
    model = LanguageModelPlusTemporal(lm_path,
                                      additional_features_size,
                                      window_size,
                                      sequence_length,
                                      num_dist_categories,
                                      num_dist_steps,
                                      main_loss_ratio=main_loss_ratio,
                                      pos_label_ratio=calculate_class_ratio(dataset)
                                      )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=25,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # save_steps=3000,
        save_total_limit=4,
        evaluation_strategy=IntervalStrategy("epoch"),
        save_strategy=IntervalStrategy("epoch"),
        # eval_steps=3000,
        logging_steps=500,
        report_to=["all"],
        label_names=["hl_labels"],  # , "objective_simple"]
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=True,
        hub_model_id=f"{lm_path.split('/')[-1]}-finetuned-highlight-detection-plus-temporal-epochs",
        # optional, will default to the name of your output directory
        hub_token=HUB_TOKEN
    )

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        args=training_args,
        train_dataset=dataset["train"],
        # train_dataset=datasets.Dataset.from_dict(ds_chk_seq_targets_padded_simple["train"][:4096]),
        eval_dataset=dataset["val"],
        # eval_dataset=datasets.Dataset.from_dict(ds_chk_seq_targets_padded_simple["val"][:64])
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    train_result = trainer.train()

    # https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data
    try:
        with open(f"netscratch/gutsche/training/log_history_{run_id}.json", "w") as out_file:
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
        with open(f"{output_dir}/final_eval.txt", "w") as out_file:
            out_file.write(str(final_eval))
    except Exception as e:
        print("cannot compute final eval")
        print(e)


def define_metrics():
    f1_metric = datasets.load_metric("f1")
    p_metric = datasets.load_metric("precision")
    r_metric = datasets.load_metric("recall")
    metrics = [f1_metric, p_metric, r_metric]
    return metrics


def preprocess_logits_for_metrics(logits, labels):
    return torch.round(torch.sigmoid(logits))


def compute_metrics(p):
    # https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    refs = p.label_ids

    preds = torch.Tensor(np.ravel(preds))
    refs = torch.tensor(np.ravel(refs))

    metrics = define_metrics()
    results = dict(ChainMap(*[m.compute(predictions=preds, references=refs) for m in metrics]))
    return results


def calculate_class_ratio(ds):
    # specifically used for binary cross entropy loss with logits
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    total_count = len(np.ravel(ds["train"]["hl_labels"]))
    pos_cat_count = sum(np.ravel(ds["train"]["hl_labels"]))
    neg_cat_count = total_count - pos_cat_count

    return neg_cat_count / pos_cat_count
