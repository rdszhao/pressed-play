import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from io import BytesIO
from PIL import Image
import boto3
import requests
import json
import base64
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from .spotifile import SpotifyClient
from inference.endpoint import model_fn, predict_fn

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
    settings.AWS_ACCESS_KEY_ID,
    settings.AWS_SECRET_ACCESS_KEY
)

import boto3

def get_dynamodb_table(table_name):
	dynamodb = boto3.resource(
						'dynamodb', 
						region_name=settings.AWS_REGION_NAME, 
						aws_access_key_id=settings.AWS_ACCESS_KEY_ID, 
						aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
	)
	return dynamodb.Table(table_name)

img_buffer = [0]

@csrf_exempt
def infer_image(request):
	if request.method == "POST":
		client = SpotifyClient(request.session['access_token'])
		image = request.FILES['image'].read()
		img_buffer[0] = image
		result = predict_fn(image, model)
		request.session['prediction'] = result
		print(result)

		features = dict(zip(TARGET_FEATURES, result))
		spotify_uri, plid = client.create_playlist(request, features)
		request.session['created_playlist_id'] = plid
		return JsonResponse({'spotifyURI': spotify_uri})

	return JsonResponse({'error': 'Only POST requests are allowed'})

@csrf_exempt
def save_playlist(request):
	if request.method == "POST":
		if 'created_playlist_id' in request.session:
			del request.session['created_playlist_id']
		return JsonResponse({'status': 'playlist saved successfully'})

	return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)

def put_to_dynamo(table, id: str, image_data: bytes, prediction: bytes, feedback: str):
	table.put_item(
		Item={
			'id': id,
			'image_data': image_data,
			'prediction': prediction,
			'feedback': feedback
		}
	)

def compress_img(image_data: bytes) -> bytes:
	image = Image.open(BytesIO(image_data))
	quality = 90
	while quality > 10:  # Ensure we don't loop indefinitely
		buffer = BytesIO()
		image.save(buffer, format="JPEG", quality=quality)
		if len(buffer.getvalue()) <= 250 * 1024:
			return buffer.getvalue()
		quality -= 5
	return buffer.getvalue()

dynamodb = boto3.resource(
	'dynamodb',
	aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
	aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
	region_name='us-east-2'
)	
table = dynamodb.Table('feedback')

@csrf_exempt
def feedback(request):
	if request.method == 'POST':
		feedback_data = json.loads(request.body)
		feedback = str(feedback_data.get('feedback', None))
		plid = str(request.session.get('created_playlist_id', None))
		prediction = np.array(request.session['prediction']).tostring()
		image_bytes = compress_img(img_buffer[0])

		put_to_dynamo(
			table,
			id=plid,
			image_data=image_bytes,
			prediction=prediction,
			feedback=feedback
		)
		img_buffer[0] = None
	return JsonResponse({'status': 'success'})