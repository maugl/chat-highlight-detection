from pprint import pprint

import numpy
import numpy as np
import json
import glob
from moviepy.editor import VideoFileClip
import re
import matplotlib.pyplot as plt
from sklearn import preprocessing

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


def average_message_lengths(cd, interval):
    if interval is None:
        interval = len(cd)
    msg_lens = message_lengths(cd)
    steps = np.arange(len(cd))[::interval]
    avg_msg_lens = np.add.reduceat(msg_lens, steps)
    return avg_msg_lens


def message_lengths(cd):
    # multiple messages per frame => calculate average
    split_re = re.compile("\n")
    msg_lengths = list()
    for frame in cd:
        if frame == "":
            msg_lengths.append(0)
        else:
            frame_msgs = [len(m) for m in split_re.split(frame)[:-1]] # remove new line message divider and split messages
            msg_lengths.append(sum(frame_msgs)/len(frame_msgs))
    return msg_lengths

def emote_density(cd, interval):
    # not quite sure how to implement yet
    pass


def emote_counts(cd):
    # not quite sure how to implement yet
    # emote_re = re.compile("\b\w+[\W|\w]+\b")
    pass


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

    if hls[-1][1] < 0:
        # highlight goes until the end
        hls[-1] = (hls[-1][0], hl.shape[0]-1)
    return hls


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

    chat = load_chat("data/final_data", file_identifier="nalcs_w1d3_TL_FLY_g2*")
    highlights = load_highlights("data/gt", file_identifier="nalcs_w1d3_TL_FLY_g2") # nalcs_w1d3_TL_FLY_g2

    matches_meta = {}
    cut = 30 * 5  # 5 sec intervals in 30 fps video, why? just because!

    for match in chat.keys():
        ch_match = chat[match]
        hl_match = highlights[match]

        hl_spans = highlight_span(hl_match)
        hl_lens = [e-s+1 for s, e in hl_spans]
        hl_count = len(hl_lens)

        cd_message_density = message_density(ch_match, interval=cut)
        cd_message_avg_len = average_message_lengths(ch_match, interval=cut)

        matches_meta[match] = {
            "highlight_spans": hl_spans,
            "highlight_lengths": hl_lens,
            "highlight_count": hl_count,
            "chat_message_density": cd_message_density
        }

    plt.plot(np.arange(len(cd_message_density)), cd_message_density / np.linalg.norm(cd_message_density), linewidth=.5)
    plt.plot(np.arange(len(cd_message_avg_len)), cd_message_avg_len / np.linalg.norm(cd_message_avg_len), linewidth=.5)
    plt.plot(np.arange(len(hl_match[::5*30])), hl_match[::cut]/4, linewidth=.5)
    plt.show()

    #pprint(matches_meta)

