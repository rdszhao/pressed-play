import numpy as np
import pandas as pd 
import boto3
from tqdm import tqdm
from config import envs

def read_from_dynamo(table, n: int) -> list:
    items = []
    last_evaluated_key = None
    
    while len(items) < n:
        if last_evaluated_key:
            response = table.scan(ExclusiveStartKey=last_evaluated_key)
        else:
            response = table.scan()
        
        items.extend(response.get('Items', []))
        
        last_evaluated_key = response.get('LastEvaluatedKey')
        if not last_evaluated_key:
            break
    return items[:n]

def clear_from_dynamo(table, items):
	for item in tqdm(items):
		table.delete_item(Key={'id': item['id']})

def stitch_feedback():
	dynamodb = boto3.resource(
		'dynamodb',
		aws_access_key_id=envs['AWS_ACCESS_KEY_ID'],
		aws_secret_access_key=envs['AWS_SECRET_ACCESS_KEY'],
		region_name='us-east-2'
	)	
	table = dynamodb.Table('feedback')
	dbdf = read_from_dynamo(table, 100)
	clear_from_dynamo(table, dbdf)
	nudata = []
	for item in dbdf:
		prediction = np.frombuffer(bytes(item['prediction']))
		image_data = bytes(item['image_data'])
		feedback = int(item['feedback'])
		nudata.append([image_data, prediction, feedback])

	df = pd.DataFrame(nudata, columns=[['cover', 'features', 'feedback']])
	df.to_parquet('feedback.parquet')

	s3 = boto3.resource(
		's3',
		aws_access_key_id=envs['AWS_ACCESS_KEY_ID'],
		aws_secret_access_key=envs['AWS_SECRET_ACCESS_KEY'],
		region_name='us-east-2'
	)	
	bucket = s3.Bucket('coverdata')
	bucket.upload_file('feedback.parquet', 'data/feedback.parquet')