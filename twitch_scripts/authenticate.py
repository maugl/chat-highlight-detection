import json

import requests
import urllib.parse

# if expires in is lower than this number in seconds then update token
REFRESH_TIME = 300


class TwitchOauth:
    oauth_token = None

    def __init__(self, cls=None, clid=None, scopes=None):
        if cls and clid and scopes:
            self.oauth_token = self.get_oauth_token(cls, clid, scopes)

    def get_oauth_token(self, cls, clid, scopes):
        scopes = "%20".join(scopes)

        auth_url = "https://id.twitch.tv/oauth2/token?" \
                   "client_id={}" \
                   "&client_secret={}" \
                   "&grant_type=client_credentials&scope={}".format(cls, clid, scopes)

        # auth_url_encoded = urllib.parse.quote(auth_url)
        response = requests.post(auth_url)
        if response.status_code == 200:
            self.oauth_token = json.loads(response.json())

        else:
            print(response.text)

        return self.oauth_token

    def check_oauth_token(self):
        return self.oauth_token["expires_in"] < REFRESH_TIME

    def refresh_oauth_token(self, cls, clid):
        new_oauth_token = self.oauth_token
        if self.check_oauth_token():
            refresh_token = self.oauth_token["refresh_token"]
            refresh_url = "https://id.twitch.tv/oauth2/token&" \
                          "refresh_token={}"\
                          "&client_id={}}"\
                          "&client_secret={}".format(refresh_token, clid, cls)

            refresh_url_encoded = urllib.parse.quote(refresh_url)
            response = requests.post(refresh_url_encoded)
            if response.status_code == 200:
                new_oauth_token = json.loads(response.json())
            self.oauth_token = new_oauth_token

        return new_oauth_token

