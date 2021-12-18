from operator import itemgetter
from pprint import pprint

import numpy
import numpy as np
import json
import glob
from moviepy.editor import VideoFileClip
import re
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy
"""FILE LOADING"""


def load_chat(chat_dir, file_identifier="*"):
    chat_files = glob.glob("{}//{}.json".format(chat_dir, file_identifier))

    chat_data = dict()

    for file_name in chat_files:
        with open(file_name, "r") as in_file:
            match_name = file_name.split("/")[-1].replace(".json", "")
            chat_data[match_name] = json.load(in_file)

    return chat_data


def load_highlights(highlight_dir, file_identifier="*"):
    chat_files = glob.glob("{}//{}.npy".format(highlight_dir, file_identifier.replace(".npy", "")))

    highlight_data = dict()

    for file_name in chat_files:
        with open(file_name, "r") as in_file:
            match_name = file_name.split("/")[-1].replace(".npy", "")
            highlight_data[match_name] = numpy.load(file_name, allow_pickle=False)

    return highlight_data


"""CHAT MEASURES"""


def message_density(cd, interval=None):
    if interval is None:
        interval = len(cd)
    msg_counts = np.asarray(message_counts(cd))
    steps = np.arange(len(cd))[::interval]
    msg_density = np.add.reduceat(msg_counts, steps)
    return msg_density


def message_counts(cd):
    msg_cnts = list()
    # individual chat messages are delimited by new line
    count_re = re.compile("\n")
    for frame in cd:
        if frame == "":
            msg_cnts.append(0)
        else:
            msg_cnts.append(len(count_re.findall(frame)))
    return msg_cnts


def average_message_lengths_chars(cd, interval):
    if interval is None:
        interval = len(cd)
    msg_lens = message_lengths_chars(cd)
    steps = np.arange(len(cd))[::interval]
    # average by interval window
    avg_msg_lens = np.add.reduceat(msg_lens, steps) / np.add.reduceat(message_counts(cd), steps)
    return avg_msg_lens


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


def message_diversity(cd, interval):
    steps = np.arange(len(cd))[::interval]
    msgs_intervals = ["\n".join(cd[steps[i-1]: steps[i]]) for i in range(len(steps))]
    msgs_tok_freqs = [token_freq(tokenize(msgs)) for msgs in msgs_intervals]
    return [normalized_entropy(token_prob(freqs)) for _, freqs in msgs_tok_freqs]


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


def emote_density(cd, interval):
    # not quite sure how to implement yet
    # maybe simple lexicon of a few emotes, maybe from emote embedding from song2021
    pass


def emote_counts(cd):
    # not quite sure how to implement yet
    # emote_re = re.compile("\b\w+[\W|\w]+\b")
    pass


def msgs_hl_non_hl(cd, hl):
    cd = np.asarray(cd)
    print(len(cd), len(hl))

    msgs_hl = cd[np.where(hl == 1)]
    msgs_non_hl = cd[np.where(hl == 0)]

    return unpack_messages(msgs_hl), unpack_messages(msgs_non_hl)


def unpack_messages(msgs):
    unpacked = []
    for m in msgs:
        ms = m.split("\n")
        unpacked.extend([m1 for m1 in ms if len(m1) > 0])



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

    return moving_aves


def plot_matches(matches):
    fig, axs = plt.subplots(len(matches.keys()))
    for i, k1 in enumerate(matches.keys()):
        ax = axs[i]
        ax.title.set_text(k1)
        for k2 in matches[k1].keys():
            if k2.startswith("chat"):
                dat = matches[k1][k2]
                ax.plot(np.arange(len(dat)), moving_avg(dat / np.linalg.norm(dat), N=5), linewidth=.5, label=k2)
            if k2 == "highlights":
                dat = matches[k1][k2]
                ax.plot(np.arange(len(dat)), dat / 10, linewidth=.5, label="highlights")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')
    plt.show()


def sanity_check():
    chat = load_chat("data/final_data", file_identifier="nalcs_w1d3_TL_FLY_g2")
    highlights = load_highlights("data/gt", file_identifier="nalcs_w1d3_TL_FLY_g2")
    # sanity check assumption: data contains info for each video frame
    # check that video has same number of frames as chat and highlights
    print("items in chat:", len(chat[list(chat.keys())[0]]))
    print("items in highlights: ", highlights[list(highlights.keys())[0]].shape[0])
    clip = VideoFileClip("data/videos/nalcs_w1d3_TL_FLY_g2.mp4")
    print("frames in video: ", clip.reader.nframes)
    print("framerate: ", clip.fps)
    # inspect some data
    print(chat[list(chat.keys())[0]][:100])
    print(highlights[list(highlights.keys())[0]][:100])
    # inspect data where there is a highlight
    highlight_ind = np.where(highlights[list(highlights.keys())[0]] == 1)[0]
    print(chat[list(chat.keys())[0]][highlight_ind[0]: highlight_ind[100]])
    print(highlights[list(highlights.keys())[0]][highlight_ind[0]: highlight_ind[100]])

    cut = 30 * 5  # 5 sec intervals in 30 fps video, why? just because!
    cd_message_counts = message_counts(ch_match)

    multichats = np.where(np.asarray(cd_message_counts) > 1)
    print("some frames with mutiple chat messages:", np.asarray(ch_match)[multichats][:10])


if __name__ == "__main__":
    # sanity_check()
    # problem with dataset: missing highlights gold standard for nalcs_w6d3_IMT_NV_g1

    file_regex = "*" # "nalcs_w1d3_TL_FLY_g*" # "nalcs_w*d3_*g1"
    chat = load_chat("data/final_data", file_identifier=file_regex)
    highlights = load_highlights("data/gt", file_identifier=file_regex) # nalcs_w1d3_TL_FLY_g2

    missing_in_hl = set(chat.keys()) - set(highlights.keys())
    missing_in_ch = set(highlights.keys()) - set(chat.keys())
    for m in missing_in_hl:
        chat.pop(m)
    for m in missing_in_ch:
        highlights.pop(m)
    print(set(chat.keys()) - set(highlights.keys()))

    matches_meta = {}
    data_totals = {
        "video_count": 0,  # number of videos in dataset
        "video_length": 0,  # total length of all videos combined
        "highlight_count": 0,  # number of total highlights
        "highlight_length": 0,  # total length of all highlights combined

        "chat_message_count": 0, # number of total chat messages in dataset
        "chat_message_count_avg_video": 0,
        "chat_message_count_hl": 0,
        "chat_message_count_non_hl": 0,
        "chat_message_count_avg_hl": 0,
        "chat_message_count_avg_non_hl": 0,
    }
    cut = 30 * 10  # 5 sec intervals in 30 fps video, why? just because!

    data_lens = []

    for match in chat.keys():
        ch_match = chat[match]
        hl_match = highlights[match]

        hl_spans = highlight_span(hl_match)

        data_lens.append((match, len(ch_match) - len(hl_match), len(hl_spans)))

        """
        hl_spans = highlight_span(hl_match)
        hl_lens = [e-s+1 for s, e in hl_spans]
        hl_count = len(hl_lens)

        cd_message_density = message_density(ch_match, interval=cut)
        cd_message_avg_len_chars = numpy.nan_to_num(average_message_lengths_chars(ch_match, interval=cut))
        cd_message_diversity = message_diversity(ch_match, interval=cut)

        matches_meta[match] = {
            "highlight_spans": hl_spans,
            "highlight_lengths": hl_lens,
            "highlight_count": hl_count,
            "highlight_avg_len": sum(hl_lens)/len(hl_lens),
            "highlights": hl_match[::cut], # not quite clean but good enough
            "chat_message_density": cd_message_density,
            "chat_message_avg_length": cd_message_avg_len_chars,
            "chat_message_diversity": cd_message_diversity
        }

        cd_messages_highlights, cd_messages_non_highlights = msgs_hl_non_hl(ch_match, hl_match) # have to check total calculation

        # total numbers over all matches
        data_totals["video_count"] += 1
        data_totals["video_length"] += (len(ch_match) / 30)  # video length in seconds (30fps)
        data_totals["highlight_count"] += hl_count
        data_totals["highlight_length"] += sum(hl_lens)

        data_totals["chat_message_count"] += sum(message_counts(ch_match))
        data_totals["chat_message_count_hl"] += len(cd_messages_highlights)
        data_totals["chat_message_count_non_hl"] += len(cd_messages_non_highlights)
        """

    # plot_matches(matches_meta)

    # pprint(data_totals)

    pprint(sorted(data_lens, key=itemgetter(1)))
    abs_missing_frames = [abs(d[1]) for d in data_lens]
    num_highlights = [d[2] for d in data_lens]
    plt.scatter(abs_missing_frames, num_highlights)
    plt.show()

    """
    for k1 in matches_meta.keys():
        for k2 in matches_meta[k1].keys():
            if type(matches_meta[k1][k2]) == np.ndarray:
                matches_meta[k1][k2] = matches_meta[k1][k2].tolist()
    with open("data/analysis/someMatches.json", "w") as out_file:
        json.dump(matches_meta, out_file)
    """


    """
    plt.plot(np.arange(len(cd_message_density)), moving_avg(cd_message_density / np.linalg.norm(cd_message_density), N=10), linewidth=.5, label="msg_density")
    plt.plot(np.arange(len(cd_message_avg_len_chars)), moving_avg(cd_message_avg_len_chars / np.linalg.norm(cd_message_avg_len_chars), N=10), linewidth=.5, label="msg_avg_len")
    plt.plot(np.arange(len(cd_message_diversity)), moving_avg(cd_message_diversity / np.linalg.norm(cd_message_diversity), N=10), linewidth=.5, label="msg_diversity")

    plt.plot(np.arange(len(hl_match[::5*30])), hl_match[::cut]/10, linewidth=.5, label="highlights")
    plt.legend(loc="upper left")
    plt.show()
    """



    #pprint(matches_meta)

