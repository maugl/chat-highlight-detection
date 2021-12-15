from pprint import pprint

import numpy
import numpy as np
import json
import glob
from moviepy.editor import VideoFileClip

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


if __name__ == "__main__":
    # sanity_check()
    chat = load_chat("data/final_data", file_identifier="nalcs_w1d3*")
    highlights = load_highlights("data/gt", file_identifier="nalcs_w1d3*") # nalcs_w1d3_TL_FLY_g2

    matches_meta = {}

    for match in chat.keys():
        ch_match = chat[match]
        hl_match = highlights[match]

        hl_spans = highlight_span(hl_match)
        hl_lens = [e-s+1 for s, e in hl_spans]
        hl_count = len(hl_lens)

        matches_meta[match] = {
            "highlight_spans": hl_spans,
            "highlight_lengths": hl_lens,
            "highlight_count": hl_count
        }

    pprint(matches_meta)



def sanity_check():
    chat = load_chat("data/final_data", file_identifier="nalcs_w1d3_TL_FLY_g2")
    highlights = load_highlights("data/gt", file_identifier="nalcs_w1d3_TL_FLY_g2")
    # sanity check assumption: data contains info for each video frame
    # check that video has same number of frames as chat and highlights
    print("items in chat:", len(chat[list(chat.keys())[0]]))
    print("items in highlights", highlights[list(highlights.keys())[0]].shape[0])
    clip = VideoFileClip("data/videos/nalcs_w1d3_TL_FLY_g2.mp4")
    print("frames in video", clip.reader.nframes)
    # inspect some data
    print(chat[list(chat.keys())[0]][:100])
    print(highlights[list(highlights.keys())[0]][:100])
    # inspect data where there is a highlight
    highlight_ind = np.where(highlights[list(highlights.keys())[0]] == 1)[0]
    print(chat[list(chat.keys())[0]][highlight_ind[0]: highlight_ind[100]])
    print(highlights[list(highlights.keys())[0]][highlight_ind[0]: highlight_ind[100]])

