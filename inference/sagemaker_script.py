import os
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

from config import envs

def sagemaker_train():
    session = boto3.Session(
        aws_access_key_id=envs['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=envs['AWS_SECRET_ACCESS_KEY'],
        region_name='us-east-2'
    )
    sagemaker_session = sagemaker.Session(boto_session=session)
    role = 'arn:aws:iam::415483028980:role/service-role/AmazonSageMaker-ExecutionRole-20231011T131034'

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
        instance_type='ml.m5.2xlarge',
        hyperparameters={},
        sagemaker_session=sagemaker_session
    )
    training_data_channel = sagemaker.inputs.TrainingInput(
        s3_data='s3://coverdata/data/train', 
        content_type='parquet'
    )
    estimator.fit({'train': training_data_channel})
    return estimator.model_data

if __name__ == '__main__':
    s3_loc = sagemaker_train()
    with open(".env", "a") as env_file:
        env_file.write(f"\nmodel_loc_s3={s3_loc}")