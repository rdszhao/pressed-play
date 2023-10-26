# PRESSED / PLAY : autoencoder music recommendations

---
There's this iconic [YouTube Video](https://youtu.be/F5WWyyYG018?si=irttbZ33tBXrvl6-&t=121) that mixes the sounds of Frank Oceans's Blonde with the collective visual art of Hayao Miyazaki and Katsuhiro Otomo's Akira. The result is a stunning collage of emotionsal sensation that really bridges the gap between sight and sound. There's one particular section that is simply sublime: a little 5 second snippet of a glider flying off into a pink and neon blue sunset sky while *Pink + White* plays in the background; a moment of perfect synergy between sight and sound. It's a particularly well curated example of something I feel that we all experience on a regular basis: that sensation of the song in your ears being perfectly matched to the vision hitting your eyes, be it a vibrant sunset or the neon colors of downtown at night, the musical and the visual enhancing each other in perfect ouroboros.

![sublimeness right here](http://moviemezzanine.com/wp-content/uploads/laputa-19-1200x648.jpg)

PRESSED / PLAY is an exploration of this cross-modal connection, leveraging the power of deep learning and the vast musical landscape thar Spotify offers. At the heart of this project is the challenge: given an image, can we deliver a playlist or a set of song selections to match the vibe of the image? Such a system isn't just tech innovation, but also a novel way for us to discover music, turning everyday visual experiences into personalized musical ones. As we delve deeper, we'll uncover the steps and considerations that went into bringing this idea to fruition. From data collection to the intricacies of model selection and finally integrating with Spotify's recommendation engine, this journey is as much about the technicalities as it is about the art of merging two sensory worlds.

---

## the problem statement - translating art

---
**"how can we capture the essence of an image and translate it into a musical experience?"**

this problem is multifaceted:

1. **cross-modal translation**: At its core, this is a cross-modal problem where the challenge lies in bridging two vastly different types of data – visual images and musical genres or attributes.

2. **subjectivity of interpretation**: The interpretation of images is highly subjective. The same image might evoke feelings of joy for one person and nostalgia for another. The music that resonates with these emotions could differ drastically.

4. **integration with existing systems**: Once we've predicted a mood, how do we translate that into a tangible outcome for the user? In our case, this means creating a playlist of songs that aligns with the predicted mood and vibe.

5. **scalability and generalization**: The solution needs to be scalable, catering to a wide array of images and diverse user tastes. It shouldn't be limited by the dataset's initial constraints.

my aim is a system that can accept any image as input and, by understanding its content and emotion, produce a curated playlist that musically resonates with the image; being not only a technical challenge but also an artistic one, opening up new avenues for personalized content discovery and a richer multimedia experience.

---

## the stack
---


```python
import os
import json
import argparse
import io
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import spotipy
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from inference.img import random_color_saturation_and_hue, random_brightness_and_contrast, random_gaussian_blur, random_gaussian_noise, random_horizontal_flip, random_rotation, random_sharpen, random_vertical_flip, random_zoom_and_crop
from inference.img import download_image_bytes, distort_encode
from inference.vae import hyperparameters
```


```python
def load_env_file(file_path):
	env = {}
	with open(file_path, 'r') as env_file:
		for line in env_file:
			line = line.strip()
			if not line:
				continue
			key, value = line.split('=', 1)
			env[key] = value.replace('"', '')
			os.environ[key] = value
	return env
```


```python
envs = load_env_file('inference/.env')
```

---
## data - collection and creation

---
before we could even get started determining what model and architecture on which to base our system, we face 2 main underlying issues:
how exactly do we build our training data, and how do we use that to generate a prediction?

### the basic inference blueprint

Luckily, the Spotify API has our solution, coming with an inbuilt endpoint, accessible via `spotipy` through `sp.recommendations`, which accepts as input, up to 5 seeding genres, and a list of target attributes with which to generate the predictions. The attributes which I've chose to work with are:


```python
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
```

which are all relatively self-explanatory, aside from `valence`, which is essentially just a measure of how "happy" a song is, in simplest terms.

let's see this process in action! continuing to use _Pink + White_ as our example, lets use the `sp.recommendations` endpoint to find some songs that are similar


```python
auth_manager = SpotifyClientCredentials(client_id=envs['SPOTIFY_CLIENT_ID'], client_secret=envs['SPOTIFY_CLIENT_SECRET'])
sp = spotipy.Spotify(auth_manager=auth_manager)
```


```python
track_id = "3xKsf9qdS1CyvXSMEid6g8"
track_details = sp.track(track_id)
artist_id = track_details['artists'][0]['id']
artist_genres = ['r-n-b', 'soul']
audio_features_result = sp.audio_features([track_id])[0]
target_features = {feature: audio_features_result[feature] for feature in AUDIO_FEATURES}
recommendations = sp.recommendations(seed_genres=artist_genres, target_audio_features=target_features, limit=10)
recommended_songs = [(track['name'], track['external_urls']['spotify']) for track in recommendations['tracks']]
recommended_songs
```




    [('Put Your Records On',
      'https://open.spotify.com/track/2nGFzvICaeEWjIrBrL2RAx'),
     ('Baby I Need Your Loving',
      'https://open.spotify.com/track/6ClsM1x4P327baDUXp2Dep'),
     ('Window Seat', 'https://open.spotify.com/track/74HYrIbnpc2xKCTenv5qKM'),
     ('Crush', 'https://open.spotify.com/track/3Txcx4jhuiTZSvhAL0WaRc'),
     ('My Girl', 'https://open.spotify.com/track/0Bd4F0Ybq3kkqj1NBS8AaY'),
     ("Don't Take It Personal",
      'https://open.spotify.com/track/5rwV5yAoPLfIjCZ64jvC2A'),
     ('1 Thing', 'https://open.spotify.com/track/1mnqraQ8oV8MX92rdOFLWW'),
     ('Fuck You', 'https://open.spotify.com/track/4ycLiPVzE5KamivXrAzGFG'),
     ("Don't Stop 'Til You Get Enough",
      'https://open.spotify.com/track/46eu3SBuFCXWsPT39Yg3tJ'),
     ('Lil Bebe', 'https://open.spotify.com/track/7esO3L3DP7bM2OOd0Rdb4W')]



and thats it! the recommendation algortihm by itself is easy enough to use, but the problem is how we retrieve the target audio features from an image

### the data problem

We have a problem. Unlike other kinds of sandbox data science projects, there are no publicly available out-of-the-box datasets we could use to drive our model. We need an extensively labelled collection of images with audio features derived from songs, but that's not something that currenty exists on the internet. So let's make one.


```python
from inference.collector import fetch_covers, fetch_track_ids, fetch_tracklist_data
```

imported are the functions that I built to accomplish, with names that are exactly as they say on the tin

- `fetch_covers` gets all the playlist `id`s and cover images of a particular user; which we by default set to `'spotify'`
- `fetch_track_ids` gets all the track `id`s in a particular playlist, which we use to get all the track`id`s for the dataset
- `fetch_tracklist_data`: gets all the audio features we want from a track, applied over all the tracks we've obtained


```python
def get_data(user='spotify', n=float('inf')):
	covers = fetch_covers(user=user, n=n)
	pl_ids, covers = (zip(*covers))
	tracklists = [fetch_track_ids(pl_id) for pl_id in tqdm(pl_ids)]
	features = [fetch_tracklist_data(t, i) if t else None for i, t in tqdm(enumerate(tracklists))]
	targets = [[list(map(track.get, AUDIO_FEATURES)) if track else None for track in tracklist] if tracklist else None for tracklist in tqdm(features)]
	return covers, targets
```


```python
covers, targets = get_data(user='spotify', n=200)
```

    50
    100
    150
    200


    100%|██████████| 200/200 [01:54<00:00,  1.74it/s]
    200it [00:54,  3.68it/s]
    100%|██████████| 200/200 [00:00<00:00, 3078.77it/s]


As seen, the returned objects are a list of URLs linking to cover images and a list (of lists) of audio features. We have 2 main possible approaches from here: we could take the average of the audio features of each list, giving us a one to (many) relationship from picture to audio features, or we could train the same image to each song individually. 

The second approach would allow us to let the model fit to a more diverse set of observations, which would allow it to generalize better, but we would then be training the model to associate the same image with a number of different audio features. To solve this problem, we can apply a set of random transformations to each image, and thus essentially ensure that every song is associated with a _unique_ image, introducing diversity to the dataset and allowing us to bootstrap outselves out of what otherwise would have been a 200 item training dataset.


```python
distortion_functions = [
    random_rotation,
    random_horizontal_flip,
    random_vertical_flip,
    random_zoom_and_crop,
    random_brightness_and_contrast,
    random_color_saturation_and_hue,
    random_gaussian_noise,
    random_gaussian_blur,
    random_sharpen,
]

def apply_random_distortions(image):
    distorted_image = image.copy()
    transformation_mask = np.random.rand(len(distortion_functions)) > 0.5
    for distortion, apply_distortion in zip(distortion_functions, transformation_mask):
        if apply_distortion:
            distorted_image = distortion(distorted_image)
    return distorted_image
```

And here, I opt to use `dask`, since stitching together the data is an extremely computationaly intensive operation and `dask` helps to speed it up by an order of magnitude.


```python
tqdm.pandas()
@delayed
def process_data(cover_url, tracklist):
    data = []
    if tracklist and cover_url:
        image_bytes = download_image_bytes(cover_url)
        if image_bytes:
            for track_features in tracklist:
                if track_features:
                    image = distort_encode(image_bytes)
                    if image:
                        data.append((image, np.array(track_features)))
    return pd.DataFrame(data, columns=['cover', 'features'])
```


```python
covers, targets
df = dd.from_pandas(pd.DataFrame(columns=['cover', 'features']), npartitions=6)
dfs = [process_data(cover, tracklist) for cover, tracklist in zip(covers, targets)]
df = dd.from_delayed(dfs)
with ProgressBar():
	df = df.compute()
df.head()
```

    [########################################] | 100% Completed | 388.22 s





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cover</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.15, 0.787, 0.621, 0.000402, 0.58, -5.009, 0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.256, 0.75, 0.733, 0.0, 0.114, -3.18, 0.0319...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.269, 0.868, 0.538, 3.34e-06, 0.0901, -8.603...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.701, 0.628, 0.523, 0.00274, 0.219, -8.307, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.0856, 0.673, 0.722, 0.0, 0.137, -3.495, 0.0...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cover</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.15, 0.787, 0.621, 0.000402, 0.58, -5.009, 0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.256, 0.75, 0.733, 0.0, 0.114, -3.18, 0.0319...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.269, 0.868, 0.538, 3.34e-06, 0.0901, -8.603...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.701, 0.628, 0.523, 0.00274, 0.219, -8.307, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.0856, 0.673, 0.722, 0.0, 0.137, -3.495, 0.0...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.0639, 0.485, 0.943, 0.179, 0.144, -4.423, 0...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.221, 0.33, 0.895, 0.737, 0.408, -3.435, 0.0...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.0148, 0.679, 0.697, 0.0119, 0.124, -5.369, ...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.296, 0.671, 0.725, 0.00022, 0.0889, -10.065...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00...</td>
      <td>[0.6, 0.618, 0.678, 0.723, 0.501, -5.199, 0.03...</td>
    </tr>
  </tbody>
</table>
<p>14403 rows × 2 columns</p>
</div>



And there it is! We've turned a 200 item dataset into over 14000 items. Now that we have our data, we can now work on the model that will bridge the gap between images and the audio features.

---

## the model - intution and process

---

### VAE with attention: the intution

Variational Autoencoders (VAEs) and attention mechanisms are separately very poewrful in various applications. In our context, this combination architecture is especially robust for cross-model inference: let's explore why

1. **data compression**: At the core of VAEs is the capability to compress complex data into a concise latent representation. This compression helps in capturing the essence of data like images or songs without much loss of information.
2. **generative tendencies**: VAEs not only encode data but can also generate new, similar data. This generative ability aids tasks like recommendation.
3. **attention's flexibility**: Attention mechanisms allow models to focus variably on different parts of the input. This dynamic focus means that the model can weigh parts of an image or song differently based on their relevance.
4. **interpretability**: One of the allures of attention mechanisms is the interpretability it brings. It offers insights into what regions of input data the model deems important.
5. **cross-modal translation**: Combining VAEs and attention provides a robust framework for translating information across different modalities, like images to music genres.

### layer-by-layer deep dive

### 1. encoder
The encoder's job in a VAE is to take input data and produce a latent representation. This latent space captures the essential features of the data.

Given input $ x $, the encoder outputs two vectors: mean $ \mu $ and variance $ \sigma^2 $. These vectors define a Gaussian distribution from which we can sample latent variables.

$$
\mu, \log \sigma^2 = \text{Encoder}(x)
$$

### 2. sampling

Using the mean and variance from the encoder, the VAE samples a point in the latent space. This sampling introduces the stochastic element of the VAE.

We sample $ z $ from the Gaussian distribution:

$$
z = \mu + \sigma \odot \epsilon
$$

where $ \epsilon $ is a random normal variable, and $ \odot $ denotes element-wise multiplication.

### 3. decoder

The decoder takes the sampled latent variable $ z $ and attempts to reconstruct the original input: $ \hat{x} $, the reconstructed input from the decoder.

$$
\hat{x} = \text{Decoder}(z)
$$

### 4. attention mechanism

Embedded within the VAE architecture, the attention mechanism allows the model to focus on specific parts of the input when constructing the latent representation. Given an input sequence $ x_1, x_2, ... x_n $, the attention scores for each element are computed as:

$$
\alpha_i = \frac{\exp(\text{score}(x_i, z))}{\sum_{j=1}^{n} \exp(\text{score}(x_j, z))}
$$

The final context vector, which is a weighted sum of the input elements, is:

$$
c = \sum_{i=1}^{n} \alpha_i x_i
$$

### 5. loss function

The VAE has a unique loss function comprising two parts: reconstruction loss and KL divergence.

- **reconstruction loss**: measures the difference between the original input and its reconstruction.
  
$$
\mathcal{L}_{recon} = \| x - \hat{x} \|^2
$$

- **kl divergence**: measures the difference between the encoded distribution and a standard normal distribution. it acts as a regularizer.

$$
\mathcal{L}_{KL} = -0.5 \sum_{i=1}^{n} (1 + \log \sigma^2 - \mu^2 - \sigma^2)
$$

The total loss is the sum of these two:

$$
\mathcal{L} = \mathcal{L}_{recon} + \mathcal{L}_{KL}
$$


### in practice

implementing the layers and mechanisms mentioned in `PyTorch`


```python
class VAEAttention(nn.Module):
	def __init__(self, image_size=(3, 128, 128), audio_feature_size=8, latent_dim=32):
		super().__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
			nn.ReLU(),
		)

		self.hidden_size = 128 * (image_size[1] // 8) * (image_size[2] // 8)

		self.attention = nn.Sequential(
			nn.Conv2d(128, 1, kernel_size=1),
			nn.Softmax(dim=2)
		)

		self.fc_mu = nn.Linear(self.hidden_size, latent_dim)

		self.fc_logvar = nn.Linear(self.hidden_size, latent_dim)

		self.decoder = nn.Linear(latent_dim, audio_feature_size)

	def encode(self, x):
		x = self.encoder(x)
		attention_weights = self.attention(x)
		x = x * attention_weights
		x = x.view(x.size(0), -1)
		mu = self.fc_mu(x)
		logvar = self.fc_logvar(x)
		return mu, logvar

	def reparameterize(self, mu, logvar):
		epsilon = torch.randn_like(mu)
		z = mu + epsilon * torch.exp(0.5 * logvar)
		return z

	def decode(self, z):
		return self.decoder(z)

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon_features = self.decode(z)
		return recon_features, mu, logvar

def vae_loss(recon_features, features, mu, logvar):
	reconstruction_loss = F.mse_loss(recon_features, features, reduction='sum')
	kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	total_loss = reconstruction_loss + kl_divergence_loss
	return total_loss
```


```python
class CustomDataset(Dataset):
	def __init__(self, data, transform=None):
		self.data = data
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		image_bytes, audio_features = self.data.iloc[idx]['cover'], self.data.iloc[idx]['features']
		image = Image.open(io.BytesIO(image_bytes))
		if self.transform:
			image = self.transform(image)
		audio = torch.tensor(audio_features, dtype=torch.float32)
		return image, audio
```


```python
image_transform = transforms.Compose([
	transforms.Resize((128, 128)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Now to train this model, I didn't do it locally on my own machine, since my resulting dataset was pretty big (76000) rows and the training would've been much more efficient on a GPU, which I don't have. Sadge. To get around this, I uploaded the code to and ran the training job on  AWS Sagemaker, which allowed me to run the training in the cloud on GPU, also allowing me to integrate S3 into how I served the model in the end.

here's the script for how I engineered that:


```python
# this is 'train.py'
def train(epochs, model_dir, train_data_directory, debug=True):
	all_files = [os.path.join(train_data_directory, file) for file in os.listdir(train_data_directory) if file.endswith('.parquet')]
	combined_df = pd.concat([pd.read_parquet(file) for file in all_files], ignore_index=True)
	if debug:
		combined_df = combined_df[:1000]
	df = combined_df
	print('data successfully obtained')

	dataset = CustomDataset(df, transform=image_transform)
	dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = VAEAttention().to(device)
	optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
	
	print('training begin')
	for epoch in range(epochs):
		for images, audio_features in dataloader:
			images = images.to(device)
			audio_features = audio_features.to(device)

			optimizer.zero_grad()
			recon_audio_features, mu, logvar = model(images)
			loss = vae_loss(recon_audio_features, audio_features, mu, logvar)
			loss.backward()
			optimizer.step()
	
		print(f"epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}")
		torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
	
	torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args = parser.parse_args()
    train(args.epochs, args.model_dir, args.train, debug=False)
```


```python
def sagemaker_train():
    session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        region_name='us-east-2'
    )
    sagemaker_session = sagemaker.Session(boto_session=session)
    role = os.environ('SAGEMAKER_ROLE')

    source_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    code_location = 's3://coverdata/model_train'

    estimator = PyTorch(
        entry_point='train.py',
        source_dir=f"../{source_dir}/",
        code_location=code_location,
        role=role,
        framework_version='1.8.1',
        py_version='py3',
        instance_count=1,
        instance_type='ml.g4dn.2xlarge',
        hyperparameters={},
        sagemaker_session=sagemaker_session
    )
    training_data_channel = sagemaker.inputs.TrainingInput(
        s3_data='s3://coverdata/data/train', 
        content_type='parquet'
    )
    estimator.fit({'train': training_data_channel})
    return estimator

if __name__ == '__main__':
    sagemaker_train()
```

Once the training finishes successfully, the model is deployed as such:


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_fn(access_id, access_key):
	model_pth = maintain_pth(access_id, access_key)
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

def maintain_pth(access_id, access_key):
	session = boto3.Session(
		aws_access_key_id=access_id,
		aws_secret_access_key=access_key,
		region_name='us-east-2'
	)
	s3 = session.resource('s3')
	s3_uri = 's3://coverdata/model/model.pth'
	bucket, model_path = s3_uri.replace("s3://", "").split("/", 1)
	local_path = '../inference/model.pth'
	s3.Bucket(bucket).download_file(model_path, local_path)
	return '../inference/model.pth'
```

---
## genre-lization - the last puzzle pieces

---

So we've collected our data, trained a model, and are now ready to make our predictions right? We have one last step. Remember how we needed to supply the target features (which we can predict with our model now, wow!) and also up to 5 genres?

Getting all of a user's genres ia relatively trivial. I just take all of their top artists, and aggregate the genres of those artists:


```python
with open('genremappings.json', 'r') as file:
    genre_mappings = json.load(file)
```


```python
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
```

You might immediately notice some things here that stand out. What are genre mappings? Why would a genre need to be mapped?

A caveat of using Spotify's `sp.recommendations` framework is Spotify only allows seeding for a very small and narrow subet of genres:


```python
sp.recommendation_genre_seeds()
```




    {'genres': ['acoustic',
      'afrobeat',
      'alt-rock',
      'alternative',
      'ambient',
      'anime',
      'black-metal',
      'bluegrass',
      'blues',
      'bossanova',
      'brazil',
      'breakbeat',
      'british',
      'cantopop',
      'chicago-house',
      'children',
      'chill',
      'classical',
      'club',
      'comedy',
      'country',
      'dance',
      'dancehall',
      'death-metal',
      'deep-house',
      'detroit-techno',
      'disco',
      'disney',
      'drum-and-bass',
      'dub',
      'dubstep',
      'edm',
      'electro',
      'electronic',
      'emo',
      'folk',
      'forro',
      'french',
      'funk',
      'garage',
      'german',
      'gospel',
      'goth',
      'grindcore',
      'groove',
      'grunge',
      'guitar',
      'happy',
      'hard-rock',
      'hardcore',
      'hardstyle',
      'heavy-metal',
      'hip-hop',
      'holidays',
      'honky-tonk',
      'house',
      'idm',
      'indian',
      'indie',
      'indie-pop',
      'industrial',
      'iranian',
      'j-dance',
      'j-idol',
      'j-pop',
      'j-rock',
      'jazz',
      'k-pop',
      'kids',
      'latin',
      'latino',
      'malay',
      'mandopop',
      'metal',
      'metal-misc',
      'metalcore',
      'minimal-techno',
      'movies',
      'mpb',
      'new-age',
      'new-release',
      'opera',
      'pagode',
      'party',
      'philippines-opm',
      'piano',
      'pop',
      'pop-film',
      'post-dubstep',
      'power-pop',
      'progressive-house',
      'psych-rock',
      'punk',
      'punk-rock',
      'r-n-b',
      'rainy-day',
      'reggae',
      'reggaeton',
      'road-trip',
      'rock',
      'rock-n-roll',
      'rockabilly',
      'romance',
      'sad',
      'salsa',
      'samba',
      'sertanejo',
      'show-tunes',
      'singer-songwriter',
      'ska',
      'sleep',
      'songwriter',
      'soul',
      'soundtracks',
      'spanish',
      'study',
      'summer',
      'swedish',
      'synth-pop',
      'tango',
      'techno',
      'trance',
      'trip-hop',
      'turkish',
      'work-out',
      'world-music']}



But the genres for our artists aren't limited to that. In fact, [Every Noise at Once](https://everynoise.com/#otherthings) has amassed a staggering collection of every genre that has been encountered in the Spotify ecosystem. Herein laid a new problem: how do I map out all of these over 5000 genres to their closest seedable genre?

I tried a number of methods, including word similarity metrics, Word2Vec / FastText similarity comparison, but in the end, none of them contain the outside contextual knowledge to properly perform a mapping like:


```python
{'musica triste brasileira': ['brazil', 'sad']}
```

As impressive as FastText is, unfortunately it can't do translation on the fly.

So I used ChatGPT. And so the `genremappings.json` file was created, allowing us to properly best map our user's top genres to genres that are seedable by Spotify.

---
# putting it all together
---

At the end of it all, here's the final code that's run in my Django app when a user uploads an image:


```python
def infer_image(request):
	client = SpotifyClient(request.session['access_token'])
	image = request.FILES['image'].read()
	result = predict_fn(image, model)
	print(result)
	features = dict(zip(TARGET_FEATURES, result))
	spotify_uri = client.create_playlist(request, features)
	return JsonResponse({'spotifyURI': spotify_uri})
```

an image is uploaded, an inference is made, a playlist is created, and we have a vibe. nice!

---
## user feedback

---

a crucial part of any subjective art technology is user feedback. to do so, I implemented a cute little "like" / "dislike" feedback system on the site that gives either a feedback value of -1 or 1, which we can integrate into our training by switching up the loss function:

original Loss:

$$
\text{loss} = \text{reconstruction loss} + \text{kl divergence}
$$

adjusted Loss with RAML:

$$
\text{loss} = \text{reconstruction loss} - \lambda \times \text{reward} + \text{kl divergence}
$$

here, $\lambda$ is a hyperparameter that controls the influence of the reward in the loss. The reward acts as a regularizer, encouraging the VAE to generate samples that are more likely to be liked by users.

in code, it looks something like


```python
def raml_loss(self, x, recon_x, mu, logvar, reward):
	BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	adjusted_loss = BCE + KLD - self.lamb * reward
	return adjusted_loss
```

every time a user submits feedback, its uploaded to a `DynamoDB` database, where whenever I want to retrain the model, the data is downloaded and sticthed into a parquet, which is then used to retrain the model with the feedback data


```python
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
```

and the training is done as such:


```python
for epoch in range(epochs):
	for data in dataloader:
		images = data[0].to(device)
		audio_features = data[1].to(device)
		optimizer.zero_grad()
		recon_audio_features, mu, logvar = model(images)

		if len(data) == 3:
			feedback = data[2].to(device)
			loss = model.raml_loss(recon_audio_features, audio_features, mu, logvar, reward=feedback)
		else:
			loss = vae_loss(recon_audio_features, audio_features, mu, logvar)

		loss.backward()
		optimizer.step()
```

___
## final remarks

---

And that's it! all the steps and intuitions behind this explained. Pretty cool, right?

And if you want to get hands on with my toy, you can visit it [here](pressedplay.rdszhao.com) at pressedplay.rdszhao.com. You just need a Spotify account. If you're interested, email me at [rdszhao@gmail.com](rdszhao@gmail.com) to try it out. Thanks for reading, and cheers!
