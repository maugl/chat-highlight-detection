import json
from argparse import ArgumentParser
from zipfile import ZipFile
import pandas as pd
import glob


def load_vid_info(info_files_path):
    vid_info = list()

    info_files = glob.glob(info_files_path)

    for f_name in info_files:
        with open(f_name, "r") as info_file:
            vid_info.extend(json.load(info_file))

    # for lookup later
    vid2ind = {v["id"]:i for i, v in enumerate(vid_info)}
    return vid_info, vid2ind


def load_vids_chat(chat_files_path, vid_info, vid2ind):
    chat_files = glob.glob(chat_files_path)


    for chat_file_name in chat_files:
        zf = ZipFile(chat_file_name)
        for f_name in zf.namelist():
            # print(f_name)if f_name.endswith("json") else None
            if f_name.endswith("json"):
                with zf.open(f_name, "r") as in_file:
                    vid_chat = json.load(in_file)
                    v_ind = vid2ind[f_name.split("/")[-1].strip(".json")]
                    chat_info = {
                        "msg_count": len(vid_chat["comments"]),
                        # might want to add more filters in future
                        "is_rerun": vid_info[v_ind]["title"].startswith("RERUN") or vid_info[v_ind]["title"].startswith("REBROADCAST")
                    }

                    vid_info[v_ind].update(chat_info)
    return vid_info


def save_vid_info(vid_info, vid_info_out):
    cols = ["id", "title", "created_at", "msg_count", "is_rerun", "duration"]

    df_vid_info = pd.DataFrame(vid_info, columns=cols)
    df_vid_info["created_at"] = pd.to_datetime(df_vid_info["created_at"])

    df_vid_info.to_csv(vid_info_out)
    

if __name__ == "__main__":
    parser = ArgumentParser(description="load information from downloaded video chat  zip files")
    parser.add_argument("info_files_path", help="path where to load the info about the videos from")
    parser.add_argument("chat_files_path", help="path where to load chat data from")
    parser.add_argument("output", help="path where to save the extracted data")

    # example run:
    # python3 extract_video_info "/home/mgut1/data/video_info/*_vids.json" "/home/mgut1/data/videos_chat/*_vids_chat.zip" /home/mgut1/data/video_info/vid_info.csv

    args = parser.parse_args()

    v_inf, v2i = load_vid_info(args.info_files_path)
    v_inf = load_vids_chat(args.chat_files_path, v_inf, v2i)
    save_vid_info(v_inf, output)
