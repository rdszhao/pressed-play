import boto3
import argparse
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask import delayed
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from collector import get_data
from img import download_image_bytes, distort_encode
from config import envs

tqdm.pandas()

@delayed
def process_data(cover_url, tracklist):
    data = []
    if tracklist and cover_url:
        image_bytes = download_image_bytes(cover_url)
        for track_features in tracklist:
            if track_features:
                image = distort_encode(image_bytes)
                data.append((image, np.array(track_features)))
    return pd.DataFrame(data, columns=['cover', 'features'])

def stitch_data(user='spotify', n=float('inf'), npartitions=6):
	covers, target = get_data(user=user, n=n)
	df = dd.from_pandas(pd.DataFrame(columns=['cover', 'features']), npartitions=npartitions)
	dfs = [process_data(cover, tracklist) for cover, tracklist in zip(covers, target)]
	df = dd.from_delayed(dfs)
	with ProgressBar():
		df = df.compute()
	return df

class DataMaker:
    def __init__(self, user, aws_access_key, aws_secret_key, region_name='us-east-2', bucket_name='coverdata', n_items=10):
        self.session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        self.s3 = self.session.resource('s3')
        self.bucket = self.s3.Bucket(bucket_name)
        self.n_items = n_items
        self.user = user

    def upload_data(self, stitched_data):
        dfs = np.array_split(stitched_data, self.n_items)
        for i, df in tqdm(enumerate(dfs, start=1)):
            parquet = df.to_parquet(index=False)
            key = f"/data/train/trainset{i}.parquet"
            self.bucket.put_object(Key=key, Body=parquet)
            print(f"successful put: {key}")

    def make(self):
        stitched_data = stitch_data(user=self.user)
        self.upload_data(stitched_data)

maker = DataMaker(envs['AWS_ACCESS_KEY_ID'], envs['AWS_SECRET_ACCESS_KEY'])
maker.make('spotify')

def make_data(user, bucket_name='coverdata', n_items=10):
    maker = DataMaker(
        user,
        envs['AWS_ACCESS_KEY_ID'],
        envs['AWS_SECRET_ACCESS_KEY'],
        bucket_name=bucket_name,
        n_items=n_items
    )
    maker.make('spotify')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="process and upload data for a specific user to dynamodb")
    parser.add_argument("user", type=str, help="name of the user to process data for")
    args = parser.parse_args()
    make_data(args.user)