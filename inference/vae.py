import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import io

hyperparameters = {
	'image_size': (3, 128, 128),
	'audio_feature_size': 8,
	'latent_dim': 32,
	'batch_size': 64,
	'learning_rate': 0.001,
	'num_epochs': 5
}

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

image_transform = transforms.Compose([
	transforms.Resize((128, 128)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])