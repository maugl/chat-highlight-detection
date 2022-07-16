import json
from collections import ChainMap
import datasets
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, IntervalStrategy, EarlyStoppingCallback

from LanguageModelPlusTemporal import LanguageModelPlusTemporal


def train_model(dataset,
                lm_path,
                output_dir,
                additional_features_size,
                window_size,
                sequence_length,
                num_dist_categories,
                num_dist_steps,
                main_loss_ratio):
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
        per_device_train_batch_size=64,
        save_steps=1000,
        save_total_limit=4,
        evaluation_strategy=IntervalStrategy("steps"),
        eval_steps=1000,
        logging_steps=500,
        report_to=["all"],
        label_names=["hl_labels"],  # , "objective_simple"]
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    train_result = trainer.train()

    # https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data
    with open(f"{trainer.args.logging_dir}/log_history.json", "w") as out_file:
        json.dump(trainer.state.log_history, out_file, indent=4) # might have to change this to copy back to storage

    trainer.save_model()

    final_eval = trainer.evaluate()
    with open(f"{output_dir}/final_eval.txt", "w") as out_file:
        out_file.write(str(final_eval))



def define_metrics():
    f1_metric = datasets.load_metric("f1")
    p_metric = datasets.load_metric("precision")
    r_metric = datasets.load_metric("recall")
    metrics = [f1_metric, p_metric, r_metric]
    return metrics


def preprocess_logits_for_metrics(logits, labels):
    return torch.round(logits)


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
