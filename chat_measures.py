import re
from collections import Counter

import numpy as np
from scipy.stats import entropy

from utils import unpack_messages


def message_density(cd, window_size=100, step_size=1):
    msg_counts = np.asarray(message_counts(cd))
    msg_density = list()

    if window_size == 1 and step_size == 1:
        return msg_counts

    for i in range(0, len(cd), step_size):
        start_ind = max(0, int(i - window_size / 2)) # window centered around frame
        end_ind = min(len(msg_counts), int(i + window_size / 2)) # window centered around frame
        msg_density.append(msg_counts[start_ind:end_ind].sum())
    return np.asarray(msg_density)


def message_counts(cd):
    msg_cnts = list()
    for frame in cd:
        if frame == "":
            msg_cnts.append(0)
        else:
            msg_cnts.append(len(unpack_messages([frame])))
    return msg_cnts


def average_message_lengths_chars(cd, window_size=100, step_size=1):
    msg_lens = np.asarray(message_lengths_chars(cd))
    avg_msg_lens = list()

    if window_size == 1 and step_size == 1:
        return msg_lens

    for i in range(0, len(cd), step_size):
        start_ind = max(0, int(i - window_size / 2)) # window centered around frame
        end_ind = min(len(msg_lens), int(i + window_size / 2)) # window centered around frame
        avg_msg_lens.append(msg_lens[start_ind:end_ind].mean())
    return np.asarray(avg_msg_lens)


def message_lengths_chars(cd):
    # multiple messages per frame => calculate average
    split_re = re.compile("\n")
    msg_lengths = list()
    for frame in cd:
        if frame == "":
            msg_lengths.append(0)
        else:
            frame_msgs = [len(m) for m in split_re.split(frame)[:-1]] # remove new line message divider and split messages
            msg_lengths.append(sum(frame_msgs))
    return msg_lengths


def message_diversity(cd, window_size=100, step_size=1):
    msgs_tok_freqs = list()

    for i in range(0, len(cd), step_size):
        start_ind = max(0, int(i - window_size / 2)) # window centered around frame
        end_ind = min(len(cd), int(i + window_size / 2)) # window centered around frame
        msgs_tok_freqs.append(token_freq(tokenize("\n".join(cd[start_ind:end_ind]))))

    return np.asarray([normalized_entropy(token_prob(freqs)) for _, freqs in msgs_tok_freqs])


def tokenize(text):
    text = re.sub("\n+", " ", text)
    return [t for t in re.split("\\s+", text) if t != ""]


def token_freq(tokens):
    freqs = Counter(tokens)
    return list(freqs.keys()), list(freqs.values())


def token_prob(freqs):
    num_tokens = len(freqs)
    return [f/num_tokens for f in freqs]


def normalized_entropy(probs):
    if len(probs) > 1:
        log_len_probs = np.log2(len(probs))
        return entropy(probs)/log_len_probs
    else:
        return 0


def emote_density(cd, emotes, window_size=100, step_size=1):
    """

    :param cd: chat data, list of frame-wise concatenated chats
    :param emotes: set of emotes to compare against
    :param window_size:
    :param step_size:
    :return:
    """
    emote_cnts = emote_counts(cd, emotes)
    emote_density = list()

    if window_size == 1 and step_size == 1:
        return emote_cnts

    for i in range(0, len(cd), step_size):
        start_ind = max(0, int(i - window_size / 2)) # window centered around frame
        end_ind = min(len(emote_cnts), int(i + window_size / 2)) # window centered around frame
        emote_density.append(emote_cnts[start_ind:end_ind].sum())
    return np.asarray(emote_density)


def emote_counts(cd, emotes):
    """
    Count emotes per frame
    :param cd: chat data, list of frame-wise concatenated chats
    :param emotes: set of emotes to compare against
    :return:
    """
    emote_cnts = list()
    for frame in cd:
        if frame == "":
            emote_cnts.append(0)
        else:
            emote_cnts.append(sum([tok in emotes for tok in tokenize("\n".join(unpack_messages([frame])))]))
    return np.asarray(emote_cnts)


def copypasta_density(cd, window_size=100, step_size=1, n_gram_length=3, threshold=30):
    """
    finds copypasta in twitch chat messages
    :param threshold: threshold for how many occurences of one ngram is considered to be copypasta
    :param cd: tokenized chat messages
    :param window_size:
    :param step_size:
    :param n_gram_length: number of n_grams to be checked
    :return: index of copy pasta messages
    """
    ngl = n_gram_length

    n_gram_counts = Counter()
    ct_ngrams = list()
    # first pass: count n_grams_total
    for frame in cd:
        if frame == "":
            ct_ngrams.append("")
            pass
        else:
            tks = tokenize("\n".join(unpack_messages([frame])))
            # n-gram generation
            ngrams = list(zip(*[([""] * ngl + tks + [""] * ngl)[i:-ngl + i] for i in range(ngl)]))
            ct_ngrams.append(ngrams)
            ngc = Counter(ngrams)
            n_gram_counts.update(ngc)

    # second pass count copy pasta n_grams per frame
    # maybe remove that if we can keep a reference to the Counter value determined above
    n_gram_indices = list()
    for frame in ct_ngrams:
        if frame == "":
            n_gram_indices.append(0)
        else:
            n_gram_indices.append(sum([n_gram_counts[n_gram] if n_gram_counts[n_gram] >= threshold else 0 for n_gram in frame]))

    n_gram_indices = np.asarray(n_gram_indices)
    cp_density = list()

    if window_size == 1 and step_size == 1:
        return n_gram_indices

    for i in range(0, len(cd), step_size):
        start_ind = max(0, int(i - window_size / 2))
        end_ind = min(len(n_gram_indices), int(i + window_size / 2))
        cp_density.append(n_gram_indices[start_ind:end_ind].sum())
    # possibly add decay if copy pasting was done further away (in terms of number of messages in between)
    return np.asarray(cp_density)