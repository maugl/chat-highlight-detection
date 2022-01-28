import re
from pprint import pprint

import analysis
from collections import Counter

def tokenize_chat(messages):
    """
    For now simple implementation, could be more sophisticated in the future
    :param messages: iterable of chat messages
    :return: iterable of tokens in messages
    """
    msgs = " ".join(messages).lower()
    regex_split = re.compile("\s+")
    return regex_split.split(msgs)



if __name__ == "__main__":
    file_regex = "nalcs*"  # "nalcs_w1d3_TL_FLY_g*" # "nalcs_w*d3_*g1"
    chat = analysis.load_chat("data/final_data", file_identifier=file_regex)
    highlights = analysis.load_highlights("data/gt", file_identifier=file_regex)  # nalcs_w1d3_TL_FLY_g2
    analysis.remove_missing_matches(chat, highlights)

    tokens = list()
    matches = list()

    token_freqs_hl = Counter()
    token_freqs_non_hl = Counter()

    for match in chat.keys():
        ch_match, hl_match = analysis.cut_same_length(chat[match], highlights[match])
        chat[match] = ch_match
        highlights[match] = hl_match

        matches.append(match)

        messages_highlights, messages_non_highlights = analysis.msgs_hl_non_hl(ch_match, hl_match)

        # highlights tokens
        chat_tokens_hl = tokenize_chat(analysis.unpack_messages(messages_highlights))
        tc_hl = Counter(chat_tokens_hl)
        token_freqs_hl.update(tc_hl)

        # non-highlights tokens
        chat_tokens_non_hl = tokenize_chat(analysis.unpack_messages(messages_non_highlights))
        tc_non_hl = Counter(chat_tokens_non_hl)
        token_freqs_non_hl.update(tc_non_hl)

    token_freq_total = Counter()
    token_freq_total.update(token_freqs_hl)
    token_freq_total.update(token_freqs_non_hl)

    print("highlights:")
    pprint(token_freqs_hl.most_common(n=100))
    print("non-highlights:")
    pprint(token_freqs_non_hl.most_common(n=100))
    print("total:")
    pprint(token_freqs_hl.most_common(n=100))
