import glob
import json
import math
import random
from pprint import pprint

import numpy as np

from utils import highlight_span, msgs_hl_non_hl
from chat_measures import message_counts, emote_density, message_diversity, copypasta_density, \
    average_message_lengths_chars, message_density


class ChatHighlightData:
    def __init__(self,
                 chat_dir="data/final_data",
                 highlight_dir="data/gt",
                 emote_dir="data/emotes",
                 frame_rate=30,
                 window=1,
                 step=1):
        # file directories
        self.chat_dir = chat_dir
        self.highlight_dir = highlight_dir
        self.emote_dir = emote_dir

        # parameters for computation of measures
        self.window = window
        self.step = step
        self._frame_rate = frame_rate

        # data structures
        self.highlights = dict()
        self.chat = dict()
        self.matches_meta = None
        self.data_totals = None
        # self.chat_measures = dict()
        self.emotes = None

    CHAT_MEASURES = {
        "message_density": message_density,
        "average_message_lengths_chars": average_message_lengths_chars,
        "message_diversity": message_diversity,
        "emote_density": emote_density,
        "copypasta_density": copypasta_density
    }

    def load_data(self, file_identifier="*", load_random=None, random_state=None):
        chat = load_chat(self.chat_dir, file_identifier=file_identifier, load_random=load_random, random_state=random_state)
        highlights = load_highlights(self.highlight_dir, file_identifier=file_identifier)  # nalcs_w1d3_TL_FLY_g2

        remove_missing_matches(chat, highlights)

        for match in chat.keys():
            ch_match, hl_match = cut_same_length(chat[match], highlights[match])
            chat[match] = ch_match
            highlights[match] = hl_match

        self.highlights = highlights
        self.chat = chat

    def load_emotes(self, emote_dir=None, file_identifier="*_emotes.txt"):
        if emote_dir is not None:
            self.emote_dir = emote_dir
        self.emotes = load_emotes(self.emote_dir, file_identifier)

    def get_highlight_data(self, unscaled=False):
        if unscaled:
            return self.highlights
        else:
            return {match: v[::self.step] for match, v in self.highlights.items()}

    def set_window_step(self, window, step):
        if self.window != window or self.step != step:
            self.window = window
            self.step = step
            """
            for measure in self.chat_measures:
                self._compute_chat_measure(measure)
            """

    def _compute_chat_measure(self, measure_name):
        measure = self.CHAT_MEASURES[measure_name]
        vals = dict()
        precomputed_message_density = None
        for match, chat in self.chat.items():
            if measure_name == "emote_density" or measure_name == "copypasta_density":
                if precomputed_message_density is None:
                    precomputed_message_density = self.get_chat_measure("message_density")
                if measure_name == "emote_density":
                    if self.emotes is not None:
                        vals[match] = measure(chat, self.emotes, self.window, self.step) / precomputed_message_density[match]
                    else:
                        raise Exception("No emotes loaded")
                if measure_name == "copypasta_density":
                    vals[match] = measure(chat, self.window, self.step) / precomputed_message_density[match]
            else:
                vals[match] = measure(chat, self.window, self.step)

        return vals

    def get_chat_measure(self, measure_name):
        return self._compute_chat_measure(measure_name)

    def _compute_match_meta_data(self):
        self.matches_meta = dict()
        for match, hl in self.highlights.items():
            hl_spans = highlight_span(hl)
            hl_lens = [e - s + 1 for s, e in hl_spans]

            self.matches_meta[match] = {
                "highlight_spans": hl_spans,
                "highlight_lengths": hl_lens,
                "highlight_count": len(hl_lens),
                "highlight_avg_len": sum(hl_lens) / len(hl_lens) if 0 < len(hl_lens) else 0,
            }

    def get_match_meta_data(self, match=None):
        if self.matches_meta is None:
            self._compute_match_meta_data()

        if match is not None:
            return self.matches_meta[match]
        else:
            return self.matches_meta

    def _compute_data_totals(self):
        self.data_totals = {
            "video_count": 0,  # number of videos in dataset
            "video_length_secs": 0,  # total length of all videos combined
            "highlight_count": 0,  # number of total highlights
            "highlight_length_secs": 0,  # total length of all highlights combined
            "highlight_min_len_frames": math.inf,  # minimum highlight length in frames
            "highlight_max_len_frames": 0,  # maximum highlight length in frames

            "chat_message_count": 0,  # number of total chat messages in dataset
            "chat_message_count_avg_video": 0,  # avg number of chat messages per video
            "chat_message_count_hl": 0,  # number of total messages in all highlight segments
            "chat_message_count_non_hl": 0,  # number of total messages in all non-highlight segments
            "chat_message_count_avg_hl": 0  # avg number of messages per highlight segment
        }

        for match in self.chat.keys():
            ch_match = self.chat[match]
            hl_match = self.highlights[match]

            match_meta = self.get_match_meta_data(match)
            hl_lens = match_meta["highlight_lengths"]
            hl_count = len(hl_lens)

            # total numbers over all matches
            self.data_totals["video_count"] += 1
            self.data_totals["video_length_secs"] += len(ch_match) / self._frame_rate  # total video length in seconds
            self.data_totals["highlight_count"] += hl_count
            self.data_totals["highlight_length_secs"] += sum(hl_lens) / self._frame_rate  # total highlight length in seconds
            self.data_totals["highlight_min_len_frames"] = min(self.data_totals["highlight_min_len_frames"],
                                                               min(hl_lens) if hl_lens else math.inf)
            self.data_totals["highlight_max_len_frames"] = max(self.data_totals["highlight_max_len_frames"],
                                                               max(hl_lens) if hl_lens else 0)

            self.data_totals["chat_message_count"] += sum(message_counts(ch_match))

            cd_messages_highlights, cd_messages_non_highlights = msgs_hl_non_hl(ch_match, hl_match)
            self.data_totals["chat_message_count_hl"] += len(cd_messages_highlights)
            self.data_totals["chat_message_count_non_hl"] += len(cd_messages_non_highlights)
        # aggregations over all matches / highligths
        self.data_totals["chat_message_count_avg_video"] = self.data_totals["chat_message_count"] / self.data_totals["video_count"]
        self.data_totals["chat_message_count_avg_hl"] = self.data_totals["chat_message_count_hl"] / self.data_totals["highlight_count"]
        self.data_totals["highlight_length_proportion"] = self.data_totals["highlight_length_secs"] / self.data_totals["video_length_secs"]
        self.data_totals["highlight_message_count_proportion"] = self.data_totals["chat_message_count_hl"] / self.data_totals["chat_message_count"]

    def get_data_totals(self):
        if self.data_totals is None:
            self._compute_data_totals()
        return self.data_totals

    def set_frame_rate(self, frame_rate=30):
        # TODO recompute all measures
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
            highlight_data[match_name] = np.load(file_name, allow_pickle=False)

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

    for m in missing_in_hl:
        cd.pop(m)
    for m in missing_in_ch:
        hd.pop(m)
    assert len(set(cd.keys()) - set(hd.keys())) == 0 # double check

    return "missing in highlights:\t", missing_in_hl, "missing in chat:\t", missing_in_ch


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



if __name__ == "__main__":
    chd = ChatHighlightData()
    file_regex = "nalcs_w1d3_TL_FLY_g*" # "nalcs*g[13]" "nalcs_w1d3_TL_FLY_g*" "nalcs_w*d3_*g1"
    chd.load_data(file_identifier=file_regex)
    chd.load_emotes()
    # pprint(chd.get_data_totals())
    # pprint(chd.get_match_meta_data())

    for m in chd.CHAT_MEASURES:
        pprint(chd.get_chat_measure(m))
        print([len(v) for v in chd.get_chat_measure(m).values()])

    pprint(chd.get_highlight_data())
    print([len(v) for v in chd.get_highlight_data().values()])

    chd.set_window_step(window=1500, step=50)
    pprint(chd.get_chat_measure("message_density"))
    print([len(v) for v in chd.get_chat_measure("message_density").values()])
    pprint(chd.get_highlight_data())
    print([len(v) for v in chd.get_highlight_data().values()])
