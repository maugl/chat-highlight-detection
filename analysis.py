import numpy
import numpy as np
import json
import glob


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


if __name__ == "__main__":
    chat = load_chat("data/final_data", file_identifier="nalcs_w1d3_TL_FLY_g3")
    highlights = load_highlights("data/gt", file_identifier="nalcs_w1d3_TL_FLY_g3")

    print(highlights[list(chat.keys())[0]].shape)
    print(highlights[list(highlights.keys())[0]].shape)

