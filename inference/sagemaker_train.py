import os
import boto3
import argparse
import sagemaker
from sagemaker.pytorch import PyTorch

from config import envs

def sagemaker_train(mode):
    session = boto3.Session(
        aws_access_key_id=envs['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=envs['AWS_SECRET_ACCESS_KEY'],
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
        s3_data=f"s3://coverdata/data/{mode}", 
        content_type='parquet'
    )
    estimator.fit({'train': training_data_channel})
    return estimator.model_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    if args.mode == 'feedback':
        from inference.dynamo import stitch_feedback
        stitch_feedback()
    sagemaker_train(args.mode)
    # s3_loc = sagemaker_train()
    # with open(".env", "a") as env_file:
    #     env_file.write(f"\nmodel_loc_s3={s3_loc}")