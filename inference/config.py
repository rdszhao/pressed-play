import os 

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

envs = load_env_file('.env')