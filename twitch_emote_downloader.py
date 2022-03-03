import json

from credentials import client_id
import requests

from twitch_authentication import authenticate


def global_emotes():
    bearer_token, _ = authenticate()
    headers = {"Authorization": f"Bearer {bearer_token}",
             "Client-ID": client_id
             }
    resp = requests.get('https://api.twitch.tv/helix/chat/emotes/global', headers=headers)

    with open("data/emotes/twitch_emotes.json", "w") as out_file:
        json.dump(resp.json(), out_file)


def list_emotes():
    emote_data = None
    with open("data/emotes/twitch_emotes.json", "r") as in_file:
        emote_data = json.load(in_file)

    emote_names = set()
    for em in emote_data["data"]:
        emote_names.add(em["name"])
        print(em["name"])

    with open("data/emotes/twitch_emotes.txt", "w") as out_file:
        out_file.write("\n".join(list(sorted(emote_names))))

    print(len(emote_data["data"]))


if __name__ == "__main__":
    # global_emotes()
    list_emotes()
