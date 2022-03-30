import json
from argparse import ArgumentParser
from zipfile import ZipFile
import pandas as pd
import glob
from datetime import datetime
import os


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


def extract_messages(chat_files_path, out_dir, vid_info, vid2ind):
    chat_files = glob.glob(chat_files_path)
    num_files_extracted = 0
    for chat_file_name in chat_files:
        zf = ZipFile(chat_file_name)
        for f_name in zf.namelist():            
            if f_name.endswith("json"):
                with zf.open(f_name, "r") as in_file:
                    vid_chat = json.load(in_file)
                    video_id = f_name.split("/")[-1].strip(".json")
                    with open(f"{out_dir}/{video_id}.txt", "w") as out_file:
                        out_file.write("\n".join([msg["message"]["body"] for msg in vid_chat["comments"]]))
                    num_files_extracted += 1
            if i%25 == 0:
                print(f"{datetime.now().strftime('%Y/%m/%d_%H:%M:%S')}: extracted {num_files_extracted} chat files.")
                        

def make_corpus_dir(corpus_dir):
    if not os.path.exists(corpus_dir):
        os.makedirs(corpus_dir)


def save_vid_info(vid_info, vid_info_out):
    cols = ["id", "title", "created_at", "msg_count", "is_rerun", "duration"]

    df_vid_info = pd.DataFrame(vid_info, columns=cols)
    df_vid_info["created_at"] = pd.to_datetime(df_vid_info["created_at"])

    df_vid_info.to_csv(vid_info_out)
    

if __name__ == "__main__":
    parser = ArgumentParser(description="load information from downloaded video chat  zip files")
    parser.add_argument("-i", "--info_files_path", help="path where to load the info about the videos from")
    parser.add_argument("-c", "--chat_files_path", help="path where to load chat data from")
    parser.add_argument("-o", "--output", help="path where to save the extracted data")
    parser.add_argument("-m", "--mode", choices=["vid_info", "chat_messages"], help="Whch action to perform, if 'vid_info', a csv file with message counts and video information is created. If 'chat_messages', all the messages are extracted into text files, one message per line, one file per video and written.")

    # example run (chat messages):
    # python3 extract_video_info -i "/home/mgut1/data/video_info/*_vids.json" -c "/home/mgut1/data/videos_chat/*_vids_chat.zip" -o /home/mgut1/data/videos_chat/corpus -m chat_messages
    # test run (chat messages):
    # python3 extract_video_info -i "/home/mgut1/data/video_info/lolpacific_vids.json" -c "/home/mgut1/data/videos_chat/*_vids_chat.zip" -o /home/mgut1/data/videos_chat/corpus -m chat_messages

    args = parser.parse_args()
    
    # prepare general video information
    v_inf, v2i = load_vid_info(args.info_files_path)
    
    if args.mode == "vid_info":
        v_inf = load_vids_chat(args.chat_files_path, v_inf, v2i)
        save_vid_info(v_inf, args.output)
    if args.mode == "chat_messages":
        print(f"{datetime.now().strftime('%Y/%m/%d_%H:%M:%S')}: extracting chat messages...")
        make_corpus_dir(args.output)
        extract_messages("../data/videos_chat/*_vids_chat.zip", args.output, v_inf, v2i)
