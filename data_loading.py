import glob
import json
import random

import numpy
import numpy as np

from analysis import ch_match
from chat_measures import message_counts, emote_density, message_diversity, copypasta_density, \
    average_message_lengths_chars, message_density


class ChatHighlightData:
    def __init__(self,
                 chat_dir=None,
                 highlight_dir=None,
                 emote_dir=None,
                 scale=1,
                 frame_rate=30):
        # file directories
        self.chat_dir = chat_dir
        self.highlight_dir = highlight_dir
        self.emote_dir = emote_dir

        # parameters for computation of measures
        self.scale = scale
        self._frame_rate = frame_rate

        # data structures
        self.highlights = dict()
        self.chat = dict()
        self.matches_meta = dict()
        self.emotes = None


    CHAT_MEASURES = {
        "message_density": message_density,
        "average_message_lengths_chars": average_message_lengths_chars,
        "message_diversity": message_diversity,
        "emote_density": emote_density,
        "copypasta_density": copypasta_density
    }

    def load_data(self, file_identifier="*", load_random=None, random_state=None):
        chat = load_chat("data/final_data", file_identifier=file_regex)
        highlights = load_highlights("data/gt", file_identifier=file_regex)  # nalcs_w1d3_TL_FLY_g2
        self.emotes = load_emotes("data/emotes", "*_emotes.txt")

    def compute_chat_measure(self, measure_name, window, step):
        pass

    def compute_meta_data(self, self, ):

    def set_frame_rate(self, frame_rate=30):
        # TODO recompute all measures and
        pass

def load_chat(chat_dir, file_identifier="*", load_random=None, random_state=None):
    chat_files = glob.glob("{}//{}.json".format(chat_dir, file_identifier))

    if load_random:
        random.seed(random_state)
        chat_files = random.sample(chat_files, load_random)

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


def load_emotes(data_dir, file_identifier="*.txt"):
    emote_files = glob.glob("{}//{}.txt".format(data_dir, file_identifier.replace(".txt", "")))
    emotes = set()
    for file_name in emote_files:
        with open(file_name, "r") as in_file:
            for line in in_file.readlines():
                emotes.add(line.strip("\n"))

    return emotes


def remove_missing_matches(cd, hd):
    missing_in_hl = set(cd.keys()) - set(hd.keys())
    missing_in_ch = set(hd.keys()) - set(cd.keys())

    print("missing in highlights:\t", missing_in_hl)
    print("missing in chat:\t", missing_in_ch)

    for m in missing_in_hl:
        cd.pop(m)
    for m in missing_in_ch:
        hd.pop(m)
    assert len(set(cd.keys()) - set(hd.keys())) == 0 # double check


def cut_same_length(cd, hd, cut_where="end"):
    """
    Cut gold standard and chat data to same lengths. In the dataset the data has differing lengths.
    :param cd: chat data
    :param hd: highlight (gold standard) data
    :param cut_where: 'end': cut off the end of the shorter data, 'start': cut off beginning of shorter data
    :return: cut chat data, cut highlight data, both to same lengths
    """

    min_len = min(len(cd), len(hd))

    if cut_where == "end":
        cd_cut = cd[:min_len]
        hd_cut = hd[:min_len]
    elif cut_where == "start":
        cd_cut = cd[len(cd) - min_len:]
        hd_cut = hd[len(hd) - min_len:]

    return cd_cut, hd_cut
