import re
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import MinMaxScaler

import analysis
from collections import Counter
import pandas as pd


def tokenize_chat(messages):
    """
    For now simple implementation, could be more sophisticated in the future
    :param messages: iterable of chat messages
    :return: iterable of tokens in messages
    """
    msgs = " ".join(messages).lower()

    regex_split = re.compile("\s+")
    return regex_split.split(msgs)


def remove_stopwords(tokens):
    """
    removing stopwords from an iterable of strings using nltk stopwords english
    :param tokens:
    :return:
    """
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in stop_words]


def detect_copypasta(ct, n_gram_length=3):
    """
    finds copypasta in twitch chat messages
    :param ct: tokenized chat messages
    :param n_gram_length: number of n_grams to be checked
    :return: index of copy pasta messages
    """
    ngl = n_gram_length
    # general n-gram generation
    n_grams = list(zip(*[(ct + [""] * ngl)[i:-ngl + i] for i in range(ngl)]))
    n_gram_counts = Counter(n_grams)
    copy_index = [n_gram_counts[n_gram] for n_gram in n_grams]

    # possibly add threshold
    # possibly add decay if copy pasting was done further away (in terms of number of messages in between)
    return copy_index

def decay(x, strength):
    """
    https://en.wikipedia.org/wiki/Exponential_decay
    :param x:
    :return:
    """
    return np.exp(-strength*x)


if __name__ == "__main__":
    file_regex = "nalcs_w1d3_TL_FLY_g1"  # "nalcs_w1d3_TL_FLY_g*" # "nalcs_w*d3_*g1"
    chat = analysis.load_chat("data/final_data", file_identifier=file_regex)
    highlights = analysis.load_highlights("data/gt", file_identifier=file_regex)  # nalcs_w1d3_TL_FLY_g2
    emotes = {em.lower() for em in analysis.load_emotes("data/emotes", "*_emotes.txt")}

    analysis.remove_missing_matches(chat, highlights)

    tokens = list()
    matches = list()

    token_freqs_hl = Counter()
    token_freqs_non_hl = Counter()

    msgs_hl = list()
    msgs_non_hl = list()

    copy_pasta_inds = dict()

    for match in chat.keys():
        ch_match, hl_match = analysis.cut_same_length(chat[match], highlights[match])
        chat[match] = ch_match
        highlights[match] = hl_match

        matches.append(match)

        # copy pasta
        ch_match_tokenized = remove_stopwords(tokenize_chat(analysis.unpack_messages(ch_match)))
        tokens.append(ch_match_tokenized)
        copy_pasta_inds[match] = detect_copypasta(ch_match_tokenized)

        # split into highlights and non-highlights
        messages_highlights, messages_non_highlights = analysis.msgs_hl_non_hl(ch_match, hl_match)

        msgs_hl.extend(messages_highlights)
        msgs_non_hl.extend(messages_non_highlights)

        # highlights tokens
        chat_tokens_hl = remove_stopwords(tokenize_chat(messages_highlights))
        tc_hl = Counter(chat_tokens_hl)
        token_freqs_hl.update(tc_hl)

        # non-highlights tokens
        chat_tokens_non_hl = remove_stopwords(tokenize_chat(messages_non_highlights))
        tc_non_hl = Counter(chat_tokens_non_hl)
        token_freqs_non_hl.update(tc_non_hl)

    token_freq_total = Counter()
    token_freq_total.update(token_freqs_hl)
    token_freq_total.update(token_freqs_non_hl)

    df = pd.DataFrame()
    top_n = 100
    tt_hl_all = token_freqs_hl.most_common(n=top_n)
    df["hl_top_tok"] = [e[0] for e in tt_hl_all]
    df["hl_top_tok_cnts"] = [e[1] for e in tt_hl_all]

    tt_non_hl_all = token_freqs_non_hl.most_common(n=top_n)
    df["non_hl_top_tok"] = [e[0] for e in tt_non_hl_all]
    df["non_hl_top_tok_cnts"] = [e[1] for e in tt_non_hl_all]

    tt_total_all = token_freq_total.most_common(n=top_n)
    df["total_top_tok"] = [e[0] for e in tt_total_all]
    df["total_top_tok_cnts"] = [e[1] for e in tt_total_all]

    df["total_top_emotes"] = [e[0] for e in token_freq_total.most_common(n=len(token_freq_total)-1) if e[0] in emotes][:top_n]
    df["total_top_emotes_cnts"] = [e[1] for e in token_freq_total.most_common(n=len(token_freq_total)-1) if e[0] in emotes][:top_n]

    df.to_csv("data/analysis/tokenCountsNoStopWordsPlusEmotes.csv")
    print(df.head())


    # distinct messages
    msgs_dist_hl = set(msgs_hl)
    msgs_dist_non_hl = set(msgs_non_hl)

    print(f"total_num_messages: {len(msgs_non_hl) + len(msgs_hl)}\ndistinct_num_messages: {len(msgs_dist_hl | msgs_dist_non_hl)}\n"
          f"total_num_messages_hl: {len(msgs_hl)}\ndistinct_num_messages_hl: {len(msgs_dist_hl)}\n"
          f"total_num_messages_non_hl: {len(msgs_non_hl)}\ndistinct_num_messages_non_hl: {len(msgs_dist_non_hl)}\n")

    m = matches[0]
    t = tokens[0]
    print(f"some copy pasta on {m}")
    """
    inds = np.flatnonzero(copy_pasta_inds[m] == np.max(copy_pasta_inds[m]))
    print([np.asarray(t)[i-10:i+10] for i in inds])
    """
    cpi = np.asarray(copy_pasta_inds[m])
    cpi[cpi < 20] = 0
    cpi[cpi.nonzero()] = 1
    plt.plot(range(len(copy_pasta_inds[m])), cpi.reshape(-1, 1))
    plt.show()
