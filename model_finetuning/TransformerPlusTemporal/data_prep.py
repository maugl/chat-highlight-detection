from sklearn.preprocessing import StandardScaler
from transformers import RobertaTokenizerFast
import numpy as np
from tqdm import tqdm
import datasets
from utils import moving_avg
from chat_measures import message_density
from data_loading import ChatHighlightData
from hub_token import HUB_TOKEN


def prepare_data(chat_directory,
                 highlight_directory,
                 additional_features,
                 additional_features_args,
                 chunking_columns,
                 tokenizer_name,
                 max_input_len,
                 train_identifier,
                 val_identifier,
                 test_identifier,
                 seed,
                 ds_intermediate):
    if ds_intermediate:
        data = datasets.load_dataset(ds_intermediate, use_auth_token=HUB_TOKEN)
    else:
        print("data loading")
        data = load_data(ch_dir=chat_directory, hl_dir=highlight_directory,
                         train_identifier=train_identifier, val_identifier=val_identifier, test_identifier=test_identifier)
        print("tokenization")
        data = tokenize(data, tokenizer_name)
        print("temporal features")
        data = add_temporal_features(data, additional_features, additional_features_args)
        print("oversample")
        train_oversampled = over_sample_binary(data["train"])
        data["train"] = train_oversampled
        print("copying highlights raw")
        copy_args = {"cname": "highlights", "cname_new": "highlights_raw"}
        data = data.map(copy_column, fn_kwargs=copy_args)
        lm_name = tokenizer_name.split('/')[-1][:10]
        data.push_to_hub(f"Epidot/private_fuetal2017_highlights_temporal_preprocessed_{lm_name}_oversampled_intermediate", private=True, token=HUB_TOKEN)

    print("data chunking")
    data = dataset_add_chunk_ids(data)
    # chunking_columns = ['highlights', 'input_ids', 'attention_mask', 'message_density_scaled']
    data = dataset_into_chunks(data, chunking_columns)
    print("data restructuring")
    data = restructure_data_for_model(data, tokenizer_name=tokenizer_name, pad_to=max_input_len)
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


# === BALANCE DATASET ===
def over_sample_binary(ds):
    label = np.asarray(ds["highlights"])
    class_counts = (abs(label.size - label.sum()).astype(int), label.sum().astype(int))
    smaller_class = np.argmin(class_counts)

    print(class_counts, smaller_class)

    ratio = abs((len(label) - class_counts[smaller_class]) / (class_counts[smaller_class]) - 1)
    print(ratio)
    smlclss_inds = np.where(label == smaller_class)[0]
    target = round(class_counts[smaller_class] * ratio)

    new_data = datasets.Dataset.from_dict({k: np.repeat(v, ratio, axis=0) for k, v in ds[smlclss_inds].items()})
    print("new_data_done")
    new_data_remainder = datasets.Dataset.from_dict({k: np.asarray(v) for k, v in ds[smlclss_inds[:target - len(new_data["highlights"])]].items()})
    print("new_data_remainder done")

    return datasets.concatenate_datasets([ds, new_data, new_data_remainder])


# === ADDITIONAL FEATURES ===
def add_temporal_features(dataset, additional_features, additional_features_args):
    for feature in additional_features:
        feature_func = None
        try:
            feature_func = AVAILABLE_FEATURE_FUNCTIONS[feature]
        except KeyError:
            print(feature, "not supported")
            exit(1)
        args = additional_features_args[feature]
        # compute feature
        dataset = datasets.DatasetDict(
            {
                ds: dataset[ds].add_column(name=feature, column=feature_func(dataset[ds], **args)) for ds in dataset
            })

        # scale feature
        dataset = scale_temporal_measure(feature, dataset)

    return dataset


def scale_temporal_measure(column_name, dataset):
    # maybe we want to save the scaler to use on future dataset
    scaler = StandardScaler()
    ds_messages_chunked_tempfeat_scld = datasets.DatasetDict()
    ds_messages_chunked_tempfeat_scld["train"] = dataset["train"].add_column(
        name=f"{column_name}_scaled", column=scaler.fit_transform(
            np.asarray(dataset["train"][column_name]).reshape(-1, 1)).ravel())
    ds_messages_chunked_tempfeat_scld["val"] = dataset["val"].add_column(
        name=f"{column_name}_scaled", column=scaler.transform(
            np.asarray(dataset["val"][column_name]).reshape(-1, 1)).ravel())
    ds_messages_chunked_tempfeat_scld["test"] = dataset["test"].add_column(
        name=f"{column_name}_scaled", column=scaler.transform(
            np.asarray(dataset["test"][column_name]).reshape(-1, 1)).ravel())
    return ds_messages_chunked_tempfeat_scld


def msg_dens_dataset(ds, window_size, step_size, mvg_avg_N):
    print("msg_dens_dataset", ds)
    match_name = ds["match_name"]
    prev = ds["match_name"][0]
    prev_cut = 0
    msg_dens = list()
    messages = ds["messages_split"]
    for i in range(len(match_name)):
        val = match_name[i]
        if prev != val:
            md = moving_avg(message_density(messages[prev_cut:i], window_size=window_size, step_size=step_size),
                            N=mvg_avg_N)
            msg_dens.extend(md)
            prev_cut = i
        prev = val
    msg_dens.extend(moving_avg(message_density(messages[prev_cut:i + 1], window_size=window_size, step_size=step_size),
                               N=mvg_avg_N))
    return msg_dens


# === CHUNKING ===
def copy_column(example, cname, cname_new):
    example[cname_new] = example[cname]
    return example

def dataset_add_chunk_ids(ds):
    ds_messages_tok_count = ds.map(
        lambda examples: {"tok_count": [len(ex) for ex in examples["input_ids"]]}, batched=True)
    ds_messages_chunked = datasets.DatasetDict(
        {ds: ds_messages_tok_count[ds].add_column(name="chunk_id", column=chunk_dataset(ds_messages_tok_count[ds])) for
         ds in ds_messages_tok_count})
    return ds_messages_chunked


def chunk_dataset(ds, max_chunk_len=32):
    cur_chunk_len = 0
    chunk_ids = list()
    tok_counts = ds["tok_count"]
    match_names = ds["match_name"]
    prev_match = match_names[0]

    def add_chunk_id(cids):
        try:
            cids.append(cids[-1])
        except IndexError:
            # should only happen with first entry
            cids.append(0)

    for i in range(len(tok_counts)):
        tc = tok_counts[i]
        mn = match_names[i]
        # ignore empty sequences where token count is 2 (only bos and eos)
        # will be removed later
        if tc < 3:
            add_chunk_id(chunk_ids)
            continue

        if cur_chunk_len + tc > max_chunk_len:
            # single frames which are too long
            # get their own id and will be truncated later
            chunk_ids.append(chunk_ids[-1] + 1)
            cur_chunk_len = tc
        elif mn != prev_match:
            # cut at match boundary
            chunk_ids.append(chunk_ids[-1] + 1)
            cur_chunk_len = tc
            prev_match = mn
        else:
            add_chunk_id(chunk_ids)
            cur_chunk_len += tc
    return chunk_ids


def pre_calculate_chunk_borders(cids):
    chunk_borders = dict()
    prev = cids[0]
    for i, val in enumerate(cids):
        if prev != val:
            chunk_borders[prev] = i
        prev = val
    if prev not in chunk_borders:
        chunk_borders[prev] = i
    return chunk_borders


def generate_frame_ids_from_chunks(ds,
                                   window_size=2,
                                   context_size=(1, 1),  # before, after
                                   step_size=2):
    chunk_ids = np.asarray(ds["chunk_id"])
    chunks = list(sorted(set(ds["chunk_id"])))

    sequence_length = window_size + sum(context_size)
    # check if last sequence ends exactly on the last frame
    # or last frame is within last sequence
    # starting a new chunk every step_size
    # window_size also plays a role in how many sequences we can fit
    last_seq = int(len(chunks) % step_size != 0)
    num_sequences = int(len(chunks) / step_size) + last_seq
    num_sequences += 1  # plus one because of the first half-sequence

    chunk_borders_pre = pre_calculate_chunk_borders(chunk_ids)
    sequence_ids = np.zeros((num_sequences, window_size + sum(context_size) + 1), dtype=int)
    # first half-sequence
    # overhang at beginning and end may be a problem for more than one sequence
    hanging_chunks = list(range(min(chunks), min(chunks) + sum(context_size)))
    sequence_ids[0][-len(hanging_chunks):] = [chunk_borders_pre[i] for i in hanging_chunks]
    sequence_ids[0][-len(hanging_chunks) - 1] = 0
    sequence_ids[0][:-(len(hanging_chunks) + 1)] = -1

    for i in range(0, num_sequences - 1 - last_seq):
        curr_seq_chunk_ids = chunks[i * step_size:i * step_size + sequence_length]
        sequence_ids[i + 1][0] = chunk_borders_pre[
            chunks[i * step_size - 1]]  # start one chunk border earlier to include the first chunk of the series
        try:
            sequence_ids[i + 1][1:] = [chunk_borders_pre[c] for c in curr_seq_chunk_ids]
        except ValueError:
            # in case we overhang on last sequence(s) we can add empty chunks
            chk_brds = [chunk_borders_pre[c] for c in curr_seq_chunk_ids]
            # make chunk "empty" by adding in last index multiple times
            sequence_ids[i + 1][1:] = chk_brds + [chk_brds[-1]] * (sequence_length - len(chk_brds))
    # last sequence if it
    if last_seq > 0:
        sequence_ids[-1][:len(chunks) - (num_sequences - 2) * step_size] = [chunk_borders_pre[c] for c in chunks[(num_sequences - 2) * step_size:]]  # last sequence (may be half empty)

    sequence_ids[1][0] = 0  # hard coded solution for wrapping back around the list with 1st element
    # if window_size is larger, this may affect more than one element
    # better: check if indices are out of order. Set highger ones to 0
    return sequence_ids


def dataset_into_chunks(dataset, cols):
    return datasets.DatasetDict(
        {ds: sequence_chunk_data(dataset[ds], cols) for ds in dataset})


def process_attmask_inputids(seq):
    ret_arr = list()
    for arr in seq:
        # ret_arr.append(list())
        try:
            for elm in arr:
                if len(elm) > 2:
                    ret_arr.extend(elm)
        except TypeError:
            ret_arr.extend(arr)
    return ret_arr


def mean_over_sequences_rounded(seq):
    return [round(e) for e in mean_over_sequences(seq)]


def mean_over_sequences(seq):
    ret_seq = list()
    for i, arr in enumerate(seq):
        try:
            ret_seq.append(sum(arr) / len(arr))
        except ZeroDivisionError:
            # happens for first and last sequence which may be only partially filled
            ret_seq.append(0)
    return ret_seq


def sequence_chunk_data(ds, cols):
    """
    aggregate each column in a specific way depending on the content of the column

    :param ds:
    :param cols:
    :return:
    """
    ds_new_data = dict()

    seq_ids = generate_frame_ids_from_chunks(ds)
    for c in cols:
        data = ds[c]
        print(c)

        # select correct aggregation function
        if c in AVAILABLE_CHUNKING_FUNCTIONS:
            f = AVAILABLE_CHUNKING_FUNCTIONS[c]
        else:
            f = None
        data_new = list()
        for sid in tqdm(seq_ids, desc=c):
            intervals = zip(sid[:-1], sid[1:])
            if f:
                # do a chunking transformation
                data_new.append(f([data[i:j] for i, j in intervals]))
            else:
                # no transformation
                data_new.append([data[i:j] for i, j in intervals])
        ds_new_data[c] = data_new
    return datasets.Dataset.from_dict(ds_new_data)


# === DS RESTRUCTURING ===
def restructure_data_for_model(ds, tokenizer_name, pad_to):
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    ds = ds.map(lambda examples: compute_labels(examples))
    ds = ds.map(lambda examples: compute_additional_features(examples))
    ds = ds.map(lambda examples: pad_truncate_to_max_sequence_length(examples, tokenizer.pad_token_id, pad_to))
    return ds


def compute_labels(ex, context_size=(1, 1)):
    return {"hl_labels": np.asarray(ex["highlights"][context_size[0]: -context_size[1]], dtype=float)}

"""
            "additional_labels": np.concatenate(
                [ex["hl_dist_prev_disc"], ex["hl_dist_next_disc"], ex["hl_dist_start_disc"], ex["hl_dist_stop_disc"]],
                axis=-1)}
"""


def compute_additional_features(ex):
    return {"additional_features": ex["message_density_scaled"]}


def pad_truncate_to_max_sequence_length(ex, pad_token_id, pad_to):
    assert len(ex["input_ids"]) == len(ex["attention_mask"])

    if len(ex["input_ids"]) >= pad_to:
        # truncate
        return {
            "input_ids": np.asarray(ex["input_ids"][:pad_to], dtype=float),
            "attention_mask": np.asarray(ex["attention_mask"][:pad_to], dtype=float)
        }
    else:
        # pad
        return {
            "input_ids": np.concatenate([ex["input_ids"], np.full((pad_to - len(ex["input_ids"])), pad_token_id)],
                                        axis=-1).astype(np.float),
            "attention_mask": np.concatenate([ex["attention_mask"], np.full((pad_to - len(ex["attention_mask"])), 0)],
                                             axis=-1).astype(np.float)
        }



# this is at the end of the document to provide global variables which can access functions
AVAILABLE_FEATURE_FUNCTIONS = {
    "message_density": msg_dens_dataset
}

AVAILABLE_CHUNKING_FUNCTIONS = {
    "highlights": mean_over_sequences_rounded,
    "attention_mask": process_attmask_inputids,
    "input_ids": process_attmask_inputids,
}
# do something similar for _disc colums
for func in AVAILABLE_FEATURE_FUNCTIONS.keys():
    AVAILABLE_CHUNKING_FUNCTIONS[f"{func}_scaled"] = mean_over_sequences
