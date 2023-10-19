from config import envs
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
import time

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5

auth_manager = SpotifyClientCredentials(client_id=envs['SPOTIFY_CLIENT_ID'], client_secret=envs['SPOTIFY_CLIENT_SECRET'])
spotify = spotipy.Spotify(auth_manager=auth_manager)
# auth_manager = SpotifyClientCredentials(client_id=os.environ['SPOTIFY_CLIENT_ID'], client_secret=os.environ['SPOTIFY_CLIENT_SECRET'])
# spotify = spotipy.Spotify(auth_manager=auth_manager)

def fetch_covers(user='spotify', n=float('inf')):
	i = 0
	covers = []
	offset = 0
	limit = 50
	total_playlists = None

	while total_playlists is None or offset < total_playlists:
		playlists = spotify.user_playlists(user, offset=offset, limit=limit)
		if not playlists['items']:
			break

		for playlist in playlists['items']:
			playlist_id = playlist['id']
			cover = playlist['images'][0]['url'] if playlist['images'] else None
			covers.append([playlist_id, cover])

		print(len(covers))
		total_playlists = playlists['total']
		offset += limit
		i += limit
		if i >= n:
			break

	return covers

def fetch_track_ids(pl_id):
	tracks = spotify.playlist_tracks(pl_id, fields='items(track(id))')
	try:
		tracklist = [track['track']['id'] for track in tracks['items']] if tracks else None
		return tracklist
	except:
		return None

def fetch_tracklist_data(tracklist, i):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            features = spotify.audio_features(tracklist)
            return features
        except Exception as e:
            retries += 1
            if retries < MAX_RETRIES:
                print(f"error fetching tracklist data (retry {retries}): {e}")
                time.sleep(RETRY_SLEEP_SECONDS)
            else:
                print(f"max retries reached while fetching tracklist data: {e}")
                return None

AUDIO_FEATURES = [
	'acousticness',
	'danceability',
	'energy',
	'instrumentalness',
	'liveness',
	'loudness',
	'speechiness',
	'valence'
]

def get_data(user='spotify', n=float('inf')):
	covers = fetch_covers(user=user, n=n)
	pl_ids, covers = (zip(*covers))
	tracklists = [fetch_track_ids(pl_id) for pl_id in tqdm(pl_ids)]
	features = [fetch_tracklist_data(t, i) if t else None for i, t in tqdm(enumerate(tracklists))]
	targets = [[list(map(track.get, AUDIO_FEATURES)) if track else None for track in tracklist] if tracklist else None for tracklist in tqdm(features)]
	return covers, targets