from pprint import pprint

import numpy as np
import json
# from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math

from chat_measures import message_density, message_counts, emote_density, copypasta_density
from data_loading import load_chat, load_highlights, load_emotes, remove_missing_matches, cut_same_length

"""HIGHLIGHT STATISTICS"""
def highlight_count(hl):
    prev = -1
    hlc = 0
    for frame in hl:
        if frame == 1 and prev < 1:
            hlc += 1
        prev = frame
    return hlc


def highlight_length(hl):
    hll = list()

    prev = -1
    for frame in hl:
        if frame == 1:
            if prev < 1:
                hll.append(1)
            else:
                hll[-1] += 1
        prev = frame
    return hll


def highlight_span(hl):
    """
    extracts beginning and end indices for each highlight in hl
    :param hl: iterable of 0 for non-hihglight frame and 1 for highlight frame
    :return: tuples of indices over all frames in hl where highlights are found
    """
    hls = list()

    prev = -1
    for i, frame in enumerate(hl):
        if frame == 1 and prev < 1:
            hls.append((i, -1))
        if frame == 0 and prev == 1:
            hls[-1] = (hls[-1][0], i-1)
        prev = frame

    if len(hls) > 0 and hls[-1][1] < 0:
        # highlight goes until the end
        hls[-1] = (hls[-1][0], hl.shape[0]-1)
    return hls


def msgs_hl_non_hl(cd, hl):
    cd = np.asarray(cd)

    msgs_hl = cd[np.where(hl == 1)]
    msgs_non_hl = cd[np.where(hl == 0)]

    return unpack_messages(msgs_hl), unpack_messages(msgs_non_hl)


"""UTILS"""


# compute moving average
# https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
# has to be improved, maybe other type of smoothing
def moving_avg(mylist, N=5):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    diff = len(mylist) - len(moving_aves)
    if diff > 0:
        tmp = [0 for i in range(diff)]
        tmp.extend(moving_aves)
        moving_aves = tmp
    return np.asarray(moving_aves)


def unpack_messages(cd):
    """
    :param cd: iterable of strings with chat messages, individual messages separated by '\n'
    :return: unpacked messages in iterable, one string per message, no empty strings returned
    """
    unpacked = []
    for m in cd:
        ms = m.split("\n")
        unpacked.extend([m1 for m1 in ms if len(m1) > 0])
    return unpacked

# TODO rewrite to be more general for allowing to plot any chat related variables without processing
def plot_matches(matches):
    fig, axs = plt.subplots(len(matches.keys()), sharex="all")
    for i, k1 in enumerate(matches.keys()): # fails if only one match is selected
        ax = axs[i]
        ax.title.set_text(k1)
        for k2 in matches[k1].keys():
            if k2.startswith("chat"):
                dat = matches[k1][k2]
                # flip moving average and MinMaxScaler
                ax.plot(np.arange(len(dat)), moving_avg(MinMaxScaler().fit_transform(dat.reshape(-1, 1)), N=1500), linewidth=.5, label=k2)
                # ax.plot(np.arange(len(dat)), MinMaxScaler().fit_transform(dat.reshape(-1, 1)), linewidth=.5, label=f"{k2} no smoothing")
            if k2 == "highlights" or k2.startswith("pred"):
                dat = matches[k1][k2]
                ax.plot(np.arange(len(dat)), dat, linewidth=.5, label=k2)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    plt.show()


if __name__ == "__main__":
    # TODO put the whole loading and calculation loop into function with parameters for which measures to output

    # sanity_check()
    # problem with dataset: missing highlights gold standard for nalcs_w6d3_IMT_NV_g1

    file_regex = "nalcs*g[13]" # "nalcs_w1d3_TL_FLY_g*" # "nalcs_w*d3_*g1"
    chat = load_chat("data/final_data", file_identifier=file_regex)
    highlights = load_highlights("data/gt", file_identifier=file_regex) # nalcs_w1d3_TL_FLY_g2
    emotes = load_emotes("data/emotes", "*_emotes.txt")

    print(len(chat))
    print(len(highlights))
    print(len(emotes))

    remove_missing_matches(chat, highlights)

    matches_meta = {}
    data_totals = {
        "video_count": 0,  # number of videos in dataset
        "video_length_secs": 0,  # total length of all videos combined
        "highlight_count": 0,  # number of total highlights
        "highlight_length_secs": 0,  # total length of all highlights combined
        "highlight_min_len_frames": math.inf,  # minimum highlight length in frames
        "highlight_max_len_frames": 0,  # maximum highlight length in frames

        "chat_message_count": 0, # number of total chat messages in dataset
        "chat_message_count_avg_video": 0, # avg number of chat messages per video
        "chat_message_count_hl": 0, # number of total messages in all highlight segments
        "chat_message_count_non_hl": 0, # number of total messages in all non-highlight segments
        "chat_message_count_avg_hl": 0 # avg number of messages per highlight segment
    }
    cut = 30 * 10  # 5, 10 sec intervals in 30 fps video, why? just because!

    data_lens = []

    for match in chat.keys():
        ch_match, hl_match = cut_same_length(chat[match], highlights[match])
        chat[match] = ch_match
        highlights[match] = hl_match

        # hl_spans = highlight_span(hl_match)

        hl_spans = highlight_span(hl_match)
        hl_lens = [e-s+1 for s, e in hl_spans]
        hl_count = len(hl_lens)

        cd_message_density = message_density(ch_match, window_size=300) # this is calculated differently than avg_len and diversity
        # cd_message_avg_len_chars = numpy.nan_to_num(average_message_lengths_chars(ch_match, interval=cut))
        # cd_message_diversity = message_diversity(ch_match, interval=cut)
        cd_emote_density = emote_density(ch_match, emotes, window_size=300)
        cd_copypasta_density = copypasta_density(ch_match, window_size=300, threshold=20, n_gram_length=5)

        matches_meta[match] = {
            "highlight_spans": hl_spans,
            "highlight_lengths": hl_lens,
            "highlight_count": hl_count,
            "highlight_avg_len": sum(hl_lens)/len(hl_lens) if 0 < len(hl_lens) else 0,
            "highlights": hl_match, # not quite clean but good enough
            "chat_message_density": cd_message_density,
            # "chat_message_avg_length": cd_message_avg_len_chars,
            # "chat_message_diversity": cd_message_diversity
            "chat_emote_density": cd_emote_density,
            "chat_copypasta_density": cd_copypasta_density
        }

        cd_messages_highlights, cd_messages_non_highlights = msgs_hl_non_hl(ch_match, hl_match)
        # how many frames of difference in length between highlight gold standard and chat data
        data_lens.append((match, len(ch_match) - len(hl_match), len(hl_spans)))

        # total numbers over all matches
        data_totals["video_count"] += 1
        data_totals["video_length_secs"] += len(ch_match) / 30  # total video length in seconds (30fps)
        data_totals["highlight_count"] += hl_count
        data_totals["highlight_length_secs"] += sum(hl_lens) / 30  # total highlight length in seconds (30fps)
        data_totals["highlight_min_len_frames"] = min(data_totals["highlight_min_len_frames"], min(hl_lens) if hl_lens else math.inf)
        data_totals["highlight_max_len_frames"] = max(data_totals["highlight_max_len_frames"], max(hl_lens) if hl_lens else 0)

        data_totals["chat_message_count"] += sum(message_counts(ch_match))
        data_totals["chat_message_count_hl"] += len(cd_messages_highlights)
        data_totals["chat_message_count_non_hl"] += len(cd_messages_non_highlights)

    # aggregations over all matches / highligths
    data_totals["chat_message_count_avg_video"] = data_totals["chat_message_count"] / data_totals["video_count"]
    data_totals["chat_message_count_avg_hl"] = data_totals["chat_message_count_hl"] / data_totals["highlight_count"]
    data_totals["highlight_length_proportion"] = data_totals["highlight_length_secs"] / data_totals["video_length_secs"]
    data_totals["highlight_message_count_proportion"] = data_totals["chat_message_count_hl"] / data_totals["chat_message_count"]

    #plot_matches(matches_meta)
    pprint(data_totals)

    """

    # show where there is a frame number discrepancy between highlight annotation and chat data
    pprint(sorted(data_lens, key=itemgetter(1)))
    abs_missing_frames = [d[1] for d in data_lens]
    num_highlights = [d[2] for d in data_lens]
    color_NALCS_LMS = ["r" if d[0].startswith("nalcs") else "b" for d in data_lens]
    plt.scatter(abs_missing_frames, num_highlights, c=color_NALCS_LMS)
    plt.title("differing length of highlight annotation and chat data, red: nalcs, blue:lms")
    plt.xlabel("number of missing frames (neg: more gold frame, pos: more chat frames)")
    plt.ylabel("number of highlights in match")
    plt.show()
    """


    for k1 in matches_meta.keys():
        for k2 in matches_meta[k1].keys():
            if type(matches_meta[k1][k2]) == np.ndarray:
                matches_meta[k1][k2] = matches_meta[k1][k2].tolist()
    with open("data/analysis/MatchesMeta_train.json", "w") as out_file:
        json.dump(matches_meta, out_file, indent=4)



    """
    plt.plot(np.arange(len(cd_message_density)), moving_avg(cd_message_density / np.linalg.norm(cd_message_density), N=10), linewidth=.5, label="msg_density")
    plt.plot(np.arange(len(cd_message_avg_len_chars)), moving_avg(cd_message_avg_len_chars / np.linalg.norm(cd_message_avg_len_chars), N=10), linewidth=.5, label="msg_avg_len")
    plt.plot(np.arange(len(cd_message_diversity)), moving_avg(cd_message_diversity / np.linalg.norm(cd_message_diversity), N=10), linewidth=.5, label="msg_diversity")

    plt.plot(np.arange(len(hl_match[::5*30])), hl_match[::cut]/10, linewidth=.5, label="highlights")
    plt.legend(loc="upper left")
    plt.show()
    """

    #pprint(matches_meta)

