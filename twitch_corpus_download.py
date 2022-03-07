import json
import math

import subprocess
import glob
import time
from datetime import datetime

from lxml import html
import requests
from credentials import client_id
from twitch_authentication import authenticate

from argparse import ArgumentParser


def lol_twitch_channels_from_liquipedia_scraper():
    # Liquipedia S-Tier / A-Tier
    top_level = ["https://liquipedia.net/leagueoflegends/S-Tier_Tournaments", "https://liquipedia.net/leagueoflegends/A-Tier_Tournaments"]
    stream_links = set()
    for tier in top_level:
        tier_resp = requests.get(tier)
        tl_page = html.fromstring(tier_resp.content)
        tournament_tables = tl_page.cssselect("div.divCell.Tournament.Header > b > a")
        for tt in tournament_tables:
            tourn_resp = requests.get(f"https://liquipedia.net{tt.get('href')}")
            tourn_page = html.fromstring(tourn_resp.content)
            streams = tourn_page.cssselect("#Streams")
            if streams:
                streams_table = streams[0].getparent().getnext()
                first_stream = streams_table.cssselect("tr:nth-child(2) > td:nth-child(2) > a")
                if first_stream:
                    stream_links = stream_links.union({first_stream[0].get("href")})
            if len(stream_links)%5 == 0:
                print(stream_links)

    with open("data/stream_links.json", "w") as out_file:
        json.dump(list(stream_links), out_file)


"""
https://dev.twitch.tv/docs/api/reference#get-channel-information

curl -X GET 'https://api.twitch.tv/helix/channels?broadcaster_id=141981764' \
-H 'Authorization: Bearer 2gbdx6oar67tqtcmt49t3wpcgycthx' \
-H 'Client-Id: wbmytr93xzw8zbg0p1izqyzzc5mbiz'
"""
def get_channel_info_from_channel_name(channel_name):
    bearer, bearer_valid_until = authenticate(scopes=["user:read:email"])
    url = "https://api.twitch.tv/helix/users"
    headers = {"Accept": "application/vnd.twitchtv.v5+json", "Authorization": f"Bearer {bearer}",
               "Client-ID": client_id}
    parameters = {
        "login": channel_name
    }

    user_resp = requests.get(url, headers=headers, params=parameters)

    return user_resp.json()["data"][0]


def get_vids_for_channel(channel_id, language, top_n=math.inf):
    """
    https://dev.twitch.tv/docs/api/reference#get-videos

    curl -X GET 'https://api.twitch.tv/helix/videos?id=335921245' \
    -H 'Authorization: Bearer 2gbdx6oar67tqtcmt49t3wpcgycthx' \
    -H 'Client-Id: uo6dggojyb8d6soh92zknwmi5ej1q2'
    """

    bearer, bearer_valid_until = authenticate()

    url = "https://api.twitch.tv/helix/videos"
    headers = {"Accept": "application/vnd.twitchtv.v5+json", "Authorization": f"Bearer {bearer}", "Client-ID": client_id}
    parameters = {
        "first": top_n if top_n <= 100 else 100,  # build in pagination if more than 100 required
        "user_id": channel_id,
        "language": language,
        "type": "archive"
    }

    cursor = None
    vid_count = 0
    all_vids = list()
    # twitch only provides 1000 top clips
    stop = False
    while vid_count < top_n and not stop:
        # control cursor and number of video ids to fetch
        # keep going until top_n video ids are downloaded or all videos ids are fetched
        if cursor is not None:
            parameters["after"] = cursor
        if top_n - vid_count < 100:
            parameters["first"] = top_n - vid_count
        # get new bearer token 15 minutes before the old one expires
        if (bearer_valid_until - datetime.now()).seconds < 30*60:
            bearer, bearer_valid_until = authenticate()
            headers["Authorization"] = f"Bearer {bearer}"

        vids_resp = requests.get(url=url, headers=headers, params=parameters)
        # print(vids_resp.json())

        # make sure the request was valid and the desired response sent
        for i in range(3):
            try:
                if vids_resp.status_code == 200:
                    vids = vids_resp.json()
                    # all videos found
                    if not vids["pagination"] or not vids["data"]:
                        stop = True
                        break

                    all_vids.extend(vids["data"])

                    vid_count = len(all_vids)
                    cursor = vids["pagination"]["cursor"]
                    break
                else:
                    vids_resp.raise_for_status()
            except requests.exceptions.HTTPError:
                print("Error:", vids_resp.status_code)
                print("try number", i + 1, "unsuccessful")
                if i == 2:
                    return all_vids  # breaks fetching of more clips
                print("wait 10 secs")
                time.sleep(10)
                # maybe add possiblity to refresh token here
            except KeyError as e:
                print(e)
                print(vids)

    return all_vids


def downloaded_videos(output_dir):
    return [fname.split("/")[-1].strip(".json") for fname in glob.glob(f"{output_dir}/*.json")]


def multi_download_video_chat(video_ids, output_dir="data/videos_chat"):
    ret_vals = list()
    existing_vids = downloaded_videos(output_dir)
    for vid in video_ids:
        if vid in existing_vids:
            continue
        ret_vals.append(download_video_chat(vid, output_dir))
        if ret_vals[-1]:
            existing_vids.append(vid)
    return ret_vals


def download_video_chat(video_id=None, output_dir="data/videos_chat"):
    if type(video_id) is str:
        sp = subprocess.run(["./TwitchDownloaderCLI", "-m", "ChatDownload", "--id", f"{video_id}", "-o",
                             f"{output_dir}/{video_id}.json"], capture_output=True)
    else:
        raise TypeError("video_id missing")

    if sp.returncode != 0:
        return False
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Discover all VODs of a given Twitch channel and"
                                        "download the chat of those videos")
    parser.add_argument("action", choices=["vids", "chat"],help="which action to perform:\n"
                                                                "vids: fetch video info for channel names\n"
                                                                "chat: download chat for fetched videos")
    parser.add_argument("output", help="path where to save the downloaded data")
    parser.add_argument("-c", "--channels", nargs="+", help="")
    parser.add_argument("-v", "--videos", nargs="+", help="Twitch video IDs to download. Only applicable if action is"
                                                          "'vids'")
    parser.add_argument("-i", "--input", help="Directory with json file(s) containing Twitch video data to "
                                                        "download. Only applicable if action is 'chat'")
    args = parser.parse_args()
    # print(multi_download_video_chat(["1304598130", "1303600567"], output_dir="data/videos_chat"))
    # lol_twitch_channels_from_liquipedia_scraper()
    # ch_names = ["lolpacific", "riotgames", "lck", "eumasters", "esl_lol", "riotlan", "garenaesports", "lec", "lpl"]

    if args.action == "vids":
        ch_names = args.channels
        ch_infos = list()
        for ch_name in ch_names:
            channel_info = get_channel_info_from_channel_name(ch_name)
            ch_infos.append(channel_info)
            ch_id = channel_info["id"]
            vids = get_vids_for_channel(ch_id, language="en")

            print(f"found {len(vids)} videos for channel '{ch_name}'")

            with open(f"data/video_info/{ch_name}_vids.json", "w") as out_vids_file:
                json.dump(vids, out_vids_file)
        with open(f"data/video_info/channels.json", "w") as out_ch_file:
            json.dump(ch_infos, out_ch_file)
    if args.action == "chat":
        if args.videos:
            multi_download_video_chat(args.videos, args.output)
        if args.path:
            channel_vids_paths = glob.glob(f"{args.input}/*.json")

            for cvp in channel_vids_paths:
                with open(cvp, "r") as in_vids_file:
                    vids = [v["id"] for v in json.load(in_vids_file)]
                    multi_download_video_chat(vids, args.output)





