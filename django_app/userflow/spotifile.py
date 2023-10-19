import io
import json
import pytz
import base64
import datetime
import numpy as np
from PIL import Image
from collections import Counter
from spotipy import Spotify

with open('genremappings.json', 'r') as file:
    genre_mappings = json.load(file)

class SpotifyClient:
    def __init__(self, token):
        self.token = token
        self.sp = Spotify(auth=token)
        self._top_artists_cache = {}
        self._genres_cache = None

    def get_top_artists(self, time_range):
        if time_range not in self._top_artists_cache:
            self._top_artists_cache[time_range] = self.sp.current_user_top_artists(limit=50, time_range=time_range)['items']
        return self._top_artists_cache[time_range]

    def map_genres(self):
        if self._genres_cache is None:
            all_genres = []
            seen_artist_ids = set()
            for term in ['short_term', 'medium_term', 'long_term']:
                artists = self.get_top_artists(term)
                for artist in artists:
                    if artist['id'] not in seen_artist_ids:
                        seen_artist_ids.add(artist['id'])
                        all_genres.extend(artist.get('genres', []))
            mappings = []
            for genre in all_genres:
                if genre in genre_mappings.keys():
                    mappings.extend(genre_mappings[genre])
            self._genres_cache = mappings
        return self._genres_cache

    def sampled_genres(self):
        mapped_genres = self.map_genres()
        mapped_genres = [genre for genre in mapped_genres if genre]
        genre_count = Counter(mapped_genres)
        genres, weights = zip(*genre_count.items())
        weights = np.array(weights) ** 1.5
        sampled_genres = np.random.choice(
            genres, size=min(5, len(genres)), replace=False, p=np.array(weights) / sum(weights)
        ).tolist()
        return sampled_genres

    def create_playlist(self, request, feature_dict):
        existing_playlist_id = request.session.get('created_playlist_id', None)
        if existing_playlist_id:
            self.sp.user_playlist_unfollow(user=request.session['user_id'], playlist_id=existing_playlist_id)
            del request.session['created_playlist_id']

        recommendations = self.sp.recommendations(seed_genres=request.session['genre_list'], limit=20, **feature_dict)
        track_uris = [track['uri'] for track in recommendations['tracks']]

        user_id = request.session['user_id']
        pst = pytz.timezone('America/Los_Angeles')
        current_time = datetime.datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S')
        description = f"made w love by ray. created on {current_time}"
        playlist = self.sp.user_playlist_create(user=user_id, name='im.playlist', public=True, description=description)
        self.sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)

        image = request.FILES['image']
        image_base64 =compress64(image)
        self.sp.playlist_upload_cover_image(playlist_id=playlist['id'], image_b64=image_base64)

        spotify_uri = f"https://open.spotify.com/embed/playlist/{playlist['id']}"
        return spotify_uri

def compress64(image, quality=85):
    img_pil = Image.open(image)
    max_size = (500, 500)
    img_pil.thumbnail(max_size)
    output = io.BytesIO()
    img_pil.save(output, format="JPEG", quality=quality)
    encoding = output.getvalue()
    image_base64 = base64.b64encode(encoding).decode('utf-8')
    return image_base64

def convert_file(file):
    image = Image.open(file)
    byte_array = io.BytesIO()
    image.save(byte_array, format='JOEG')
    return byte_array.getvalue()