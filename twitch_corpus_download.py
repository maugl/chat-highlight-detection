import tcd
import subprocess
import glob
from lxml import html
import requests

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
            if len(stream_links)%10 == 0:
                print(stream_links)


def existing_videos(output_dir):
    return glob.glob(f"{output_dir}/*.json")


def multi_download_video_chat(video_ids, output_dir="~/Downloads"):
    ret_vals = list()
    for vid in video_ids:
        ret_vals.append(download_video_chat(vid, output_dir))
    return ret_vals


def download_video_chat(video_id=None, output_dir="~/Downloads"):
    if type(video_id) is str:
        sp = subprocess.run(['tcd', '--video', video_id, '--format', 'json', '--output', output_dir], capture_output=True)
    else:
        raise TypeError("video_id missing")

    if sp.returncode != 0:
        return False
    return True


if __name__ == "__main__":
    # print(multi_download_video_chat(["1304598130", "1303600567"], output_dir="data/videos_chat"))
    lol_twitch_channels_from_liquipedia_scraper()
