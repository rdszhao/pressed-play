# %%
import torch
from inference.vae import VAEAttention, image_transform, hyperparameters
from PIL import Image
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
estimator = VAEAttention().to(device)
estimator.load_state_dict(torch.load('model.pth'))
estimator.eval()
# %%
# replace with the received imaege
img = Image.open('Unknown.jpg')
# %%

img_tensor = image_transform(img).unsqueeze(0).to(device)
with torch.no_grad():
	features = estimator(img_tensor)[0]
output = features.tolist()[0]