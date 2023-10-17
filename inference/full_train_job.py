from datamaker import make_data
from sagemaker_script import sagemaker_train

if __name__ == '__main__':
	make_data(user='spotify')
	sagemaker_train()