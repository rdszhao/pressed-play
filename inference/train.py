import os
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from vae import VAEAttention, vae_loss, CustomDataset, image_transform, hyperparameters

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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args = parser.parse_args()
    train(args.epochs, args.model_dir, args.train, debug=False)