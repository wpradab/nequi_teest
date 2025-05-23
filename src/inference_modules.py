import os
import time
import boto3
import psycopg2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gensim.models import KeyedVectors
from utils import get_tweet_embedding, inverse_map_classes


def load_env_vars():
    load_dotenv()
    return {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }


def get_db_data(env_vars, query):
    conn = psycopg2.connect(**env_vars)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def download_and_prepare_embedding(bucket_name, file_key, local_file_path, embedding_dim=50):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, file_key, local_file_path)

    num_lines = sum(1 for _ in open(local_file_path))
    fixed_path = '/tmp/glove_fixed.txt'

    with open(local_file_path, 'r') as original, open(fixed_path, 'w') as fixed:
        fixed.write(f"{num_lines} {embedding_dim}\n")
        for line in original:
            fixed.write(line)

    return KeyedVectors.load_word2vec_format(fixed_path, binary=False)


def create_payload(df, model, embedding_dim):
    X = np.vstack(df['clean_text'].apply(lambda x: get_tweet_embedding(x, model, embedding_dim)).values)
    payload = "\n".join(",".join(f"{v:.2f}" for v in row) for row in X)
    return payload


def get_latest_training_job_info(sm_client):
    response = sm_client.list_training_jobs(MaxResults=5)
    job_name = response['TrainingJobSummaries'][0]['TrainingJobName']
    response = sm_client.describe_training_job(TrainingJobName=job_name)

    return {
        'model_artifact': response['ModelArtifacts']['S3ModelArtifacts'],
        'role_arn': response['RoleArn'],
        'image_uri': response['AlgorithmSpecification']['TrainingImage']
    }


def create_and_deploy_model(sm_client, model_info):
    timestamp = str(int(time.time()))
    model_name = f"xgboost-model-from-package-{timestamp}"
    endpoint_name = f"xgboost-registry-endpoint-{timestamp}"

    create_model_response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=model_info['role_arn'],
        PrimaryContainer={
            'Image': model_info['image_uri'],
            'ModelDataUrl': model_info['model_artifact']
        }
    )

    print("Modelo creado:", create_model_response['ModelArn'])

    endpoint_config_name = endpoint_name + "-config"
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large'
        }]
    )

    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    # Esperar a que el endpoint esté listo
    while True:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print("Estado del endpoint:", status)
        if status in ['InService', 'Failed']:
            break
        time.sleep(15)

    return endpoint_name


def invoke_endpoint(runtime_client, endpoint_name, payload):
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )
    result = response['Body'].read().decode('utf-8')
    y_pred = [float(x) for x in result.strip().split('\n')]
    return y_pred


def main():
    env_vars = load_env_vars()
    query = """
    SELECT *
    FROM public.tweets_process
    WHERE created_at >= '2017-11-30'
    ORDER BY created_at DESC;
    """
    df = get_db_data(env_vars, query)

    embedding_model = download_and_prepare_embedding(
        bucket_name='embedding-tweets',
        file_key='glove.twitter.27B.50d.txt',
        local_file_path='/tmp/glove.twitter.27B.50d.txt'
    )

    payload = create_payload(df, embedding_model, embedding_dim=50)

    sm_client = boto3.client('sagemaker')
    model_info = get_latest_training_job_info(sm_client)
    endpoint_name = create_and_deploy_model(sm_client, model_info)

    runtime_client = boto3.client('sagemaker-runtime')
    y_pred = invoke_endpoint(runtime_client, endpoint_name, payload)

    result_output = pd.Series(y_pred).apply(inverse_map_classes)
    print(result_output)


if __name__ == "__main__":
    main()
