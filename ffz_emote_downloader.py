import json
import time
import requests
import json
import glob


def dl_ffz_library(start_page=1, end_page=100):
    headers = {"accept": "application/json"}
    for i in range(start_page, end_page+1):
        resp = requests.get(f"https://api.frankerfacez.com/v1/emotes?sensitive=false&high_dpi=off&page={i}&per_page=200", headers=headers)
        with open(f"data/emotes/FFZ/library/page{i}.js", "w") as out_file:
            json.dump(resp.json(), out_file)

        """
        if i%4 == 0:
            # wait for 2 seconds before the next burst of requests
            time.sleep(2)
        """

        if i % 50 == 0:
            print(f"downloaded {i} pages")
            print(f"rate limit: {int(resp.headers['RateLimit-Remaining'])}")

        if int(resp.headers["RateLimit-Remaining"]) < 30:
            time.sleep(int(resp.headers["RateLimit-Reset"]))


def compile_emote_list(path="data/emotes/FFZ/library"):
    files = glob.glob(f"{path}/*.js")
    all_emotes = list()
    for fname in files:
        with open(fname, "r") as in_file:
            emotes = json.load(in_file)
            all_emotes.extend(emotes["emoticons"])

    with open("data/emotes/FFZ/ffz_emotes_20220201.json", "w") as out_file:
        json.dump(all_emotes, out_file)

    emote_names = set()
    for em in all_emotes:
        emote_names.add(em["name"])

    with open("data/emotes/FFZ/ffz_emote_list.txt", "w") as out_file:
        out_file.write("\n".join(list(sorted(emote_names))))


if __name__ == "__main__":
    # dl_ffz_library(start_page=1172, end_page=1499)
    compile_emote_list()