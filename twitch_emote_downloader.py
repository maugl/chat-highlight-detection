import json

from credentials import client_id, client_secret
import requests


def global_emotes():
    headers={"Authorization": f"Bearer {authenticate()}",
             "Client-ID": client_id
             }
    resp = requests.get('https://api.twitch.tv/helix/chat/emotes/global', headers=headers)

    with open("data/emotes/twitch_emotes.json", "w") as out_file:
        json.dump(resp.json(), out_file)


def authenticate():
    resp = requests.post(f"https://id.twitch.tv/oauth2/token?client_id={client_id}&client_secret={client_secret}&grant_type=client_credentials")
    print(resp.json())
    # print(resp.json()["access_token"])
    # resp_val = requests.get("https://id.twitch.tv/oauth2/validate", headers={'Authorization': resp.json()["access_token"]})
    # print(resp_val.json())
    return resp.json()["access_token"]


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
