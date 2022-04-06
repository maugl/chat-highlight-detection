import torch

import transformers
from transformers import LineByLineTextDataset, RobertaForMaskedLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaTokenizerFast
from transformers import RobertaConfig

import datasets
from datasets import load_dataset

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers.pre_tokenizers import Whitespace

def load_tokenizer(path):
    tokenizer = ByteLevelBPETokenizer(
        f"{path.rstrip('/')}/vocab.json",
        f"{path.rstrip('/')}/merges.txt",
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)

    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.mask_token = "<mask>"

    return tokenizer

def load_model():
    config = RobertaConfig(
        vocab_size=30_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    model = RobertaForMaskedLM(config=config)


def load_data_set(path):
    return datasets.DatasetDict.load_from_disk(path)


def check_for_cuda():
    assert(torch.cuda.is_available())


def train(model, tokenizer, dataset, output_dir):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train']
    )

    trainer.train()

if __name__ == "__main__":
    check_for_cuda()

    model_path = "/home/mgut1/data/LMtraining/TwitchLeagueBert"
    tokenizer = load_tokenizer(model_path)
    model = load_model()
    dataset = load_data_set("/home/mgut1/data/LMtraining/corpus_grouped_dataset")

    train(model, tokenizer, dataset, model_path)
