import re
from pprint import pprint
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
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


if __name__ == "__main__":
    file_regex = "nalcs*"  # "nalcs_w1d3_TL_FLY_g*" # "nalcs_w*d3_*g1"
    chat = analysis.load_chat("data/final_data", file_identifier=file_regex)
    highlights = analysis.load_highlights("data/gt", file_identifier=file_regex)  # nalcs_w1d3_TL_FLY_g2
    analysis.remove_missing_matches(chat, highlights)

    tokens = list()
    matches = list()

    token_freqs_hl = Counter()
    token_freqs_non_hl = Counter()

    msgs_hl = list()
    msgs_non_hl = list()

    for match in chat.keys():
        ch_match, hl_match = analysis.cut_same_length(chat[match], highlights[match])
        chat[match] = ch_match
        highlights[match] = hl_match

        matches.append(match)

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

    tt_hl_all = token_freqs_hl.most_common(n=1000)
    df["hl_top_tok"] = [e[0] for e in tt_hl_all]
    df["hl_top_tok_cnts"] = [e[1] for e in tt_hl_all]

    tt_non_hl_all = token_freqs_non_hl.most_common(n=1000)
    df["non_hl_top_tok"] = [e[0] for e in tt_non_hl_all]
    df["non_hl_top_tok_cnts"] = [e[1] for e in tt_non_hl_all]

    tt_total_all = token_freq_total.most_common(n=1000)
    df["total_top_tok"] = [e[0] for e in tt_total_all]
    df["total_top_tok_cnts"] = [e[1] for e in tt_total_all]

    # df.to_csv("data/analysis/tokenCountsNoStopWords.csv")
    print(df.head())


    # distinct messages
    msgs_dist_hl = set(msgs_hl)
    msgs_dist_non_hl = set(msgs_non_hl)

    print(f"total_num_messages: {len(msgs_non_hl) + len(msgs_hl)}\ndistinct_num_messages: {len(msgs_dist_hl | msgs_dist_non_hl)}\n"
          f"total_num_messages_hl: {len(msgs_hl)}\ndistinct_num_messages_hl: {len(msgs_dist_hl)}\n"
          f"total_num_messages_non_hl: {len(msgs_non_hl)}\ndistinct_num_messages_non_hl: {len(msgs_dist_non_hl)}\n")
