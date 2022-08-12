import argparse
import os
import shutil
from os.path import exists
import json

import torch

from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from transformers import Trainer
from transformers import TrainingArguments

import datasets
from datasets import load_dataset

from FileWriterCallback import FileWriterCallback

datasets.disable_caching()

from hub_token import HUB_TOKEN


class CannotLoadTokenizer(Exception):
    pass


def train_tokenizer(ds, out_file_path):
    batch_size = 1000

    def batch_iterator():
        for i in range(0, len(ds["train"]), batch_size):
            yield ds["train"][i: i + batch_size]["text"]

    print("training tokenizer...")
    tokenizer_old = RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer_new = tokenizer_old.train_new_from_iterator(batch_iterator(), vocab_size=50000)
    tokenizer_new.save_pretrained(out_file_path)
    print("training tokenizer DONE")


def tokenize_dataset(tokenizer_files_path, ds):
    tokenizer_tok_data = load_tokenizer(tokenizer_files_path)

    def tokenize_function(examples):
        # using return_special_tokens_mask=True for optimized DataCollator later
        return tokenizer_tok_data(examples["text"], return_special_tokens_mask=True)
    print("tokenizing dataset...")
    dataset_tok = ds.map(tokenize_function, batched=True, num_proc=2, remove_columns=["text"])
    print("tokenizing dataset DONE")
    return dataset_tok


def load_tokenizer(path):
    try:
        tokenizer_roberta = RobertaTokenizerFast.from_pretrained(path, max_len=512)
    except Exception as e:
        raise CannotLoadTokenizer(e)
    return tokenizer_roberta


def load_model():
    config = RobertaConfig(
        vocab_size=50_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    return RobertaForMaskedLM(config=config)


def load_huggingface_dataset(corpus_file):
    return load_dataset('text', data_files={'train': corpus_file})


def check_for_cuda():
    assert(torch.cuda.is_available())


def train(model_train, tokenizer_train, ds, output_dir):
    """

    :param model_train:
    :param tokenizer_train:
    :param ds:
    :param output_dir:
    :return:
    """

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_train, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=64,
        save_steps=20_000,
        save_total_limit=4,
        prediction_loss_only=True,
        evaluation_strategy="steps",
        logging_steps=500,
        eval_steps=20_000,
        report_to="all",
        load_best_model_at_end=True,

        push_to_hub=True,
        hub_model_id="twitch-league-roberta-base-test",
        hub_token=HUB_TOKEN
    )

    trainer = Trainer(
        model=model_train,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3),
                   FileWriterCallback(log_file_path="/tmp/ml_train_logs/", write_interval=0, log_file_final_path="/netscratch/gutsche/logs/")]
    )

    trainer.train()

    trainer.save_model()
    try:
        trainer.save_state()
    except Exception as e:
        print("cannot save trainer state")
        print(e)

    # https://stackoverflow.com/questions/68806265/huggingface-trainer-logging-train-data
    with open(f"{output_dir.rstrip('/')}/log_history_.json", "w") as out_file:
        json.dump(trainer.state.log_history, out_file, indent=4)  # might have to change this to copy back to storage

    """
    final_eval = trainer.evaluate()
    with open(f"{output_dir}/final_eval.txt", "w") as out_file:
        out_file.write(str(final_eval))
    """


def group_dataset(ds, tker):
    print("grouping dataset...")
    grouped_ds = ds.map(
        group_texts,
        fn_kwargs={"tokenizer": tker},
        batched=True,
        batch_size=1000,
        num_proc=2
    )
    print("grouping dataset DONE")
    return grouped_ds


def group_texts(examples, tokenizer, block_size=512):
    """
    :param tokenizer:
    :param examples: DatasetDict containing fields with iterables to group
    :param block_size: maximum size of each group in items (tokens)

    :return: Each entry of examples grouped to block_size (number of tokens)
    """
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # keep remainder to add on later
    remainder = total_length % block_size
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # add remaining tokens
    if remainder > 0:
        for k in concatenated_examples.keys():
            if k == "input_ids":
                result[k].append(concatenated_examples[k][-remainder:] + ([tokenizer.pad_token_id] * (block_size-remainder)))
            else:
                result[k].append(concatenated_examples[k][-remainder:] + ([type(concatenated_examples[k][0])()] * (block_size-remainder)))

    result["labels"] = result["input_ids"].copy()
    return result


def load_data(tok_path, d_path, tokenizer):
    # load dataset from disk if it has been created before
    if exists(f"{d_path.rstrip('/')}/corpus_grouped_dataset"):
        print("loading grouped dataset from disk")
        ds_lm = datasets.DatasetDict.load_from_disk(f"{d_path.rstrip('/')}/corpus_grouped_dataset")
    else:
        # load dataset from disk if it has been created before
        if exists(f"{d_path.rstrip('/')}/corpus_tokenized_dataset"):
            print("loading tokenized dataset from disk")
            dataset_tokenized = datasets.DatasetDict.load_from_disk(f"{d_path.rstrip('/')}/corpus_tokenized_dataset")
        else:
            print("loading text dataset from disk")
            ds_text = load_huggingface_dataset(f"{d_path.rstrip('/')}/twitch_lol_combined.txt")
            dataset_tokenized = tokenize_dataset(ds=ds_text, tokenizer_files_path=tok_path)
            dataset_tokenized.save_to_disk(f"{d_path.rstrip('/')}/corpus_tokenized_dataset")
        dataset_grouped = group_dataset(dataset_tokenized, tokenizer)
        dataset_grouped.save_to_disk(f"{d_path.rstrip('/')}/corpus_grouped_dataset")
        ds_lm = dataset_grouped

    return ds_lm


def shuffle_split_dataset(ds, seed):
    return ds["train"].train_test_split(test_size=0.1, seed=seed)


def main(train_model=False, seed=42069, model_path="/tmp/model/TwitchLeagueBert", data_path="/tmp/data/", output_path=None, load_data_from_hub=None):
    """
    :param train_model: if true the model is trained, only dataset preparation otherwise
    :param seed: seed for data shuffling and model initialization
    :param model_path: where to load tokenizer from
    :param data_path: where to load data from
    :param output_path: where to save model to
    :return:
    """
    check_for_cuda()

    tokenizer_output = f"{output_path.rstrip('/')}/TwitchLeagueBert"

    try:
        os.makedirs(tokenizer_output)
    except FileExistsError as e:
        pass

    try:
        print("load tokenizer from disk")
        tokenizer = load_tokenizer(model_path)
        tokenizer.save_pretrained(tokenizer_output)
        # tokenizer_output = model_path
    except CannotLoadTokenizer as e:
        # if there is no tokenizer, train it
        print("load dataset from text file")
        ds_text = load_huggingface_dataset(f"{data_path.rstrip('/')}/twitch_lol_combined.txt")
        train_tokenizer(ds=ds_text, out_file_path=tokenizer_output)
        tokenizer = load_tokenizer(tokenizer_output)

    if load_data_from_hub:
        dataset_lm_shuffled = datasets.load_dataset(load_data_from_hub, use_auth_token=HUB_TOKEN)
    else:
        dataset_lm = load_data(tokenizer_output, data_path, tokenizer)

        dataset_lm_shuffled = shuffle_split_dataset(dataset_lm, seed=seed)
        dataset_lm_shuffled.save_to_disk(f"{data_path.rstrip('/')}/ds_mlm_training_twitch_LOL")
        dataset_lm_shuffled.push_to_hub("Epidot/twitch_lol_corpus_for_mlm_training", token=HUB_TOKEN, private=True)

    try:
        shutil.copytree(f"{data_path.rstrip('/')}/ds_mlm_training_twitch_LOL", f"{output_path}/ds_mlm_training_twitch_LOL")
    except Exception as e:
        print("cannot copy data to storage")

    if train_model:
        model = load_model()
        train(model, tokenizer, dataset_lm_shuffled, f"{output_path.rstrip('/')}/TwitchLeagueBert")


def log_progress(log):
    print(json.dumps(log))
