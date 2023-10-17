import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import requests
import base64
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from .spotifile import SpotifyClient
from inference.inference import model_fn, predict_fn

def login(request):
	scope = 'user-top-read playlist-modify-private playlist-modify-public ugc-image-upload'
	auth_url = f"https://accounts.spotify.com/authorize?response_type=code&client_id={settings.SPOTIFY_CLIENT_ID}&redirect_uri={settings.SPOTIFY_REDIRECT_URI}&scope={scope}"
	return redirect(auth_url)

def callback(request):
	auth_token = request.GET['code']
	auth_header = base64.b64encode(f"{settings.SPOTIFY_CLIENT_ID}:{settings.SPOTIFY_CLIENT_SECRET}".encode('ascii'))
	headers = {'Authorization': f"Basic {auth_header.decode('ascii')}"}
	data = {
		'grant_type': 'authorization_code',
		'code': auth_token,
		'redirect_uri': settings.SPOTIFY_REDIRECT_URI
	}
	auth_response = requests.post('https://accounts.spotify.com/api/token', headers=headers, data=data)
	request.session['access_token'] = auth_response.json()['access_token']

	user_profile_response = requests.get('https://api.spotify.com/v1/me', headers={
		'Authorization': f"Bearer {request.session['access_token']}"
	})

	if user_profile_response.status_code == 200:
		request.session['user_id'] = user_profile_response.json()['id']
	else:
		print(f"Error {user_profile_response.status_code}: {user_profile_response.content}")
		return JsonResponse({"error": "Failed to fetch user's Spotify profile"}, status=500)

	return redirect('get_genres')

def get_genres(request):
	client = SpotifyClient(request.session['access_token'])
	all_samples = client.sampled_genres()
	genres_string = ','.join(all_samples)
	request.session['genres'] = genres_string
	request.session['genre_list'] = all_samples
	return redirect('mainpage')

def shuffle_genres(request):
	client = SpotifyClient(request.session['access_token'])
	all_samples = client.sampled_genres()
	genres_string = ','.join(all_samples)
	request.session['genres'] = genres_string
	request.session['genre_list'] = all_samples
	return JsonResponse({'genre_list': all_samples})

# @login_required
def mainpage(request):
	return render(request, 'mainpage.html', {'genre_list': request.session.get('genre_list', '')})

TARGET_FEATURES = [
	'target_acousticness',
	'target_danceability',
	'target_energy',
	'target_instrumentalness',
	'target_liveness',
	'target_loudness',
	'target_speechiness',
	'target_valence'
]

model = model_fn(
    settings.MODEL_LOC_S3,
    settings.AWS_ACCESS_KEY_ID,
    settings.AWS_SECRET_ACCESS_KEY
)

@csrf_exempt
def infer_image(request):
	if request.method == "POST":
		client = SpotifyClient(request.session['access_token'])
		image = request.FILES['image'].read()
		result = predict_fn(image, model)
		print(result)
		features = dict(zip(TARGET_FEATURES, result))
		spotify_uri = client.create_playlist(request, features)
		return JsonResponse({'spotifyURI': spotify_uri})

	return JsonResponse({'error': 'Only POST requests are allowed'})

@csrf_exempt
def save_playlist(request):
	if request.method == "POST":
		if 'created_playlist_id' in request.session:
			del request.session['created_playlist_id']
		return JsonResponse({'status': 'Playlist saved successfully'})

	return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)