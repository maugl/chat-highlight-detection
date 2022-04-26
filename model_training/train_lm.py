from os.path import exists

import torch

from transformers import DataCollatorForLanguageModeling
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from transformers import Trainer
from transformers import TrainingArguments

import datasets
from datasets import load_dataset


def train_tokenizer(ds, out_file_path):
    batch_size = 1000

    def batch_iterator():
        for i in range(0, len(ds["train"]), batch_size):
            yield ds["train"][i: i + ds]["text"]

    tokenizer_old = RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer_new = tokenizer_old.train_new_from_iterator(batch_iterator(), vocab_size=50000)
    tokenizer_new.save_pretrained(out_file_path)


def tokenize_dataset(tokenizer_files_path, ds):
    tokenizer_tok_data = load_tokenizer(tokenizer_files_path)

    def tokenize_function(examples):
        # using return_special_tokens_mask=True for optimized DataCollator later
        return tokenizer_tok_data(examples["text"], return_special_tokens_mask=True)

    dataset_tok = ds.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    return dataset_tok


def load_tokenizer(path):
    tokenizer_roberta = RobertaTokenizerFast.from_pretrained(path, max_len=512)

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
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model_train,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds['train']
    )

    trainer.train(resume_from_checkpoint=True)

    trainer.save_model()


def group_dataset(ds):
    return ds.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )


def group_texts(examples, tokenizer, block_size=128):
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


def load_data(m_path, d_path):
    # load dataset from disk if it has been created before
    if exists(f"{d_path.rstrip('/')}/corpus_grouped_dataset"):
        ds_lm = datasets.DatasetDict.load_from_disk(f"{d_path.rstrip('/')}/corpus_grouped_dataset")
    else:
        # load dataset from disk if it has been created before
        if exists(f"{d_path.rstrip('/')}/corpus_tokenized_dataset"):
            dataset_tokenized = datasets.DatasetDict.load_from_disk(f"{d_path.rstrip('/')}/corpus_tokenized_dataset")
        else:
            ds_text = load_huggingface_dataset(f"{d_path.rstrip('/')}/twitch_lol_combined.txt")
            dataset_tokenized = tokenize_dataset(ds=ds_text, tokenizer_files_path=m_path)
            dataset_tokenized.save_to_disk(f"{d_path.rstrip('/')}/corpus_tokenized_dataset")
        dataset_grouped = group_dataset(dataset_tokenized)
        dataset_grouped.save_to_disk(f"{d_path.rstrip('/')}/corpus_grouped_dataset")
        ds_lm = dataset_grouped

    return ds_lm


def main():
    check_for_cuda()

    model_path = "/netscratch/gutsche/data/TwitchLeagueBert"
    data_path = "/netscratch/gutsche/data/"

    try:
        tokenizer = load_tokenizer(model_path)
    except OSError as e:
        # if there is no tokenizer, train it
        ds_text = load_huggingface_dataset(f"{data_path.rstrip('/')}/twitch_lol_combined.txt")
        train_tokenizer(ds=ds_text, out_file_path=model_path)
        tokenizer = load_tokenizer(model_path)

    dataset_lm = load_data(model_path, data_path)

    # model = load_model()

    #train(model, tokenizer, dataset_lm, model_path)


if __name__ == "__main__":
    main()
