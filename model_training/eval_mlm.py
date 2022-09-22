import argparse
import json

from transformers import DataCollatorForLanguageModeling
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import Trainer
from transformers import TrainingArguments
import datasets


def eval_model(m, ds, tok):
    trainer = load_trainer(ds, tok)
    eval_res = trainer.evaluate()
    print(eval_res)
    return eval_res


def save_evaluation_results(eval_res, path):
    with open(path.lower().strip(".json") + ".json", "w") as out_file:
        json.dump(eval_res, out_file)


def load_trainer(ds, tok):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./RoBERTa/evaluation_test",
        overwrite_output_dir=False,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=4,
        prediction_loss_only=True,
        evaluation_strategy="steps",
        eval_steps=5_000,
        report_to="all",
        per_device_eval_batch_size=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=None,
        eval_dataset=ds
    )

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate a language model on twitch chat data")
    parser.add_argument('-m', '--model_specifier', help="either huggingface model name or path to model")
    parser.add_argument('-d', '--data_path', help="path to the huggingface dataset to load")
    parser.add_argument('-r', '--results_path', help="where to save the results in json format")
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug:
        print(vars(args))
        exit(0)

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_specifier)
    model = RobertaForMaskedLM.from_pretrained(args.model_specifier)
    dataset = datasets.load_from_disk(args.data_path)

    evaluation_result = eval_model(model, dataset, tokenizer)
    save_evaluation_results(evaluation_result, args.results_path)