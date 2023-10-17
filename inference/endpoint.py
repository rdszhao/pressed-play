import os
import tarfile
import shutil
import boto3
from io import BytesIO
import torch
from inference.vae import VAEAttention, image_transform
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_fn(s3_uri, access_id, access_key):
	model_pth = maintain_pth(s3_uri ,access_id, access_key)
	model = VAEAttention().to(device)
	model.load_state_dict(torch.load(model_pth, map_location=device))
	model.eval()
	print('model init complete')
	return model

def predict_fn(input_data, model):
	image = Image.open(BytesIO(input_data))
	img_tensor = image_transform(image).unsqueeze(0).to(device)
	with torch.no_grad():
		features = model(img_tensor)[0]
	output = features.tolist()[0]
	return output

def maintain_pth(s3_uri, access_id, access_key):
	session = boto3.Session(
		aws_access_key_id=access_id,
		aws_secret_access_key=access_key,
		region_name='us-east-2'
	)
	s3 = session.resource('s3')
	bucket, model_path = s3_uri.replace("s3://", "").split("/", 1)
	local_tar_path = '../inference/model.tar.gz'
	s3.Bucket(bucket).download_file(model_path, local_tar_path)

	with tarfile.open(local_tar_path, 'r:gz') as tar:
		tar.extractall(path='../inference/')

	os.remove(local_tar_path)
	if os.path.isdir('../inference/code'):
		shutil.rmtree('../inference/code')

	return '../inference/model.pth'