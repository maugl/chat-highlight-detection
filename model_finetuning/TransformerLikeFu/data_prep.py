import datasets
import numpy as np
import torch
from torch import cuda
from transformers import RobertaModel, PreTrainedModel
import json
from data_loading import ChatHighlightData
from transformers import RobertaTokenizerFast


def prepare_data(chat_directory,
                 highlight_directory,
                 tokenizer_name,
                 train_identifier,
                 val_identifier,
                 test_identifier,
                 window_len,
                 step,
                 seed):

    print("data loading")
    data = load_data(ch_dir=chat_directory, hl_dir=highlight_directory,
                     train_identifier=train_identifier, val_identifier=val_identifier, test_identifier=test_identifier)

    print("tokenization")
    data = tokenize(data, tokenizer_name)

    print("grouping dataset")
    grouping_params = {
        "window_len": window_len,
        "step": step
    }
    grouping_batch_size = grouping_params["window_len"] * grouping_params["step"]
    data = data.map(group_dataset, batch_size=grouping_batch_size, batched=True,
                                         fn_kwargs=grouping_params, remove_columns=["messages_split", "match_name"])

    print("truncating and padding")
    tok = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    truncation_params = {"pad_token_id": tok.pad_token_id, "pad_to": 128} # 128 for comparability
    data = data.map(pad_truncate_to_max_sequence_length, fn_kwargs=truncation_params, batched=False)

    print("restructuring prediction")
    data = data.map(restructure_prediction, batched=True, remove_columns=["highlights"])
    print("oversampling dataset")
    data_oversampled = over_sample_binary(data["train"])
    data["train"] = data_oversampled

    # shuffle data
    data_shuffled = data.shuffle(seed=seed)
    return data_shuffled


# === DATA LOADING ===
def load_data(ch_dir="./final_data/", hl_dir="./gt/", train_identifier=None, val_identifier=None, test_identifier=None):
    chd_datasets = list()
    for chd in load_chat_hl_data(ch_dir, hl_dir, train_identifier=train_identifier, val_identifier=val_identifier,
                                 test_identifier=test_identifier):
        chd_datasets.append(chat_highlight_data_to_huggingface_dataset(chd))
    return create_dataset_dict(["train", "val", "test"], chd_datasets)


def load_chat_hl_data(ch_dir, hl_dir, train_identifier, val_identifier, test_identifier):
    chd_train = ChatHighlightData(chat_dir=ch_dir, highlight_dir=hl_dir, emote_dir=None, frame_rate=30)
    chd_train.load_data(file_identifier=train_identifier)
    chd_val = ChatHighlightData(chat_dir=ch_dir, highlight_dir=hl_dir, emote_dir=None, frame_rate=30)
    chd_val.load_data(file_identifier=val_identifier)
    chd_test = ChatHighlightData(chat_dir=ch_dir, highlight_dir=hl_dir, emote_dir=None, frame_rate=30)
    chd_test.load_data(file_identifier=test_identifier)

    assert (sorted(chd_train.chat.keys()) == sorted(chd_train.highlights.keys()))
    assert (sorted(chd_val.chat.keys()) == sorted(chd_val.highlights.keys()))
    assert (sorted(chd_test.chat.keys()) == sorted(chd_test.highlights.keys()))

    return chd_train, chd_val, chd_test


def chat_highlight_data_to_huggingface_dataset(chd):
    chat = list()
    highlights = list()
    match_name = list()

    for m, ch in chd.chat.items():
        hl = chd.highlights[m]
        try:
            assert (len(ch) == len(hl))
        except AssertionError:
            print("not matching lengths in:", m)
        name = [m] * len(ch)

        chat.extend(ch)
        highlights.extend(hl)
        match_name.extend(name)

    return datasets.Dataset.from_dict({"messages": chat,
                                       "highlights": highlights,
                                       "match_name": match_name
                                       })


def create_dataset_dict(names, data_sets):
    return datasets.DatasetDict({n: d for n, d in zip(names, data_sets)})


# === TOKENIZATION ===
def tokenize(ds, tokenizer_name):
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    ds_messages_split = ds.map(lambda example: split_add_bos_eos(example, tokenizer), remove_columns=["messages"])
    return ds_messages_split.map(lambda examples: tokenizer(examples['messages_split']), batched=True)


def split_add_bos_eos(example, tok):
    return {"messages_split": f"{tok.eos_token}{tok.bos_token}".join(example["messages"].rstrip("\n").split("\n"))}


# === DATASET GROUPING ===
def group_dataset(ds_batch, window_len=210, step=30):
    window_inds = list()
    for i in range(0, len(ds_batch["attention_mask"]) - window_len + step, step):
        tmp_w_inds = (i, i+window_len)
        mn = ds_batch["match_name"][tmp_w_inds[0]: tmp_w_inds[1]]
        if len(set(mn)) > 1:
            # stop at earlier index
            np.argmax(np.asarray(mn) == mn[-1])
        else:
            window_inds.append(tmp_w_inds)

    ret = dict()
    for key, val in ds_batch.items():
        ret[key] = list()
        for i_start, i_end in window_inds:
            if key == "input_ids" or key == "attention_mask" :
                ret[key].append(np.concatenate(val[i_start: i_end]))
            elif key == "highlights":
                ret[key].append(val[i_start])
            else:
                pass
        if len(ret[key]) == 0:
            del ret[key]

    return ret


# === TRUNCATION AND PADDING
def pad_truncate_to_max_sequence_length(ex, pad_token_id, pad_to):
    assert len(ex["input_ids"]) == len(ex["attention_mask"])

    if len(ex["input_ids"]) >= pad_to:
        # truncate
        return {
            # adds an additional sequence beginning token
            "input_ids": np.concatenate([[0], np.asarray(ex["input_ids"][-(pad_to-1):])]).astype(np.float),
            "attention_mask": np.concatenate([[1], np.asarray(ex["attention_mask"][-(pad_to-1):])]).astype(np.float)
        }
    else:
        # pad
        return {
            "input_ids": np.concatenate([ex["input_ids"], np.full((pad_to - len(ex["input_ids"])), pad_token_id)],
                                        axis=-1).astype(np.float),
            "attention_mask": np.concatenate([ex["attention_mask"], np.full((pad_to - len(ex["attention_mask"])), 0)],
                                             axis=-1).astype(np.float)
        }


# === FORMAT PREDICTION ===
def restructure_prediction(ds_batch):
  ret = list()
  for ex in ds_batch["highlights"]:
    # use int here for classification loss instead of regression loss
    ret.append([float(ex)])
  return {"label": ret}


def over_sample_binary(ds):
    label = np.asarray(ds["labels"])
    class_counts = (abs(label.size - label.sum()).astype(int), label.sum().astype(int))
    smaller_class = np.argmin(class_counts)

    print(class_counts, smaller_class)

    ratio = abs((len(label) - class_counts[smaller_class]) / (class_counts[smaller_class]) - 1)
    print(ratio)
    smlclss_inds, _ = np.where(label == smaller_class)
    print(smlclss_inds.dtype)
    target = round(class_counts[smaller_class] * ratio)

    new_data = datasets.Dataset.from_dict({k: np.repeat(v, ratio, axis=0) for k, v in ds[smlclss_inds].items()})
    new_data_remainder = datasets.Dataset.from_dict(
        {k: np.asarray(v) for k, v in ds[smlclss_inds[:target - len(new_data["labels"])]].items()})

    return datasets.concatenate_datasets([ds, new_data, new_data_remainder])