from datetime import datetime, timedelta

import requests

from twitch_scripts.credentials import client_id, client_secret


def authenticate(scopes=None):
    scope_param = f"&scope={' '.join(scopes)}" if scopes else ""
    resp = requests.post(f"https://id.twitch.tv/oauth2/token?client_id={client_id}&client_secret={client_secret}&grant_type=client_credentials" + scope_param)
    auth = resp.json()
    return auth["access_token"], datetime.now() + timedelta(seconds=auth["expires_in"])