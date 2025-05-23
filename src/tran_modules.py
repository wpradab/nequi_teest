import os
import json
import boto3
import psycopg2
import sagemaker
import numpy as np
import pandas as pd
from datetime import date
from dotenv import load_dotenv
from gensim.models import KeyedVectors
from sagemaker import get_execution_role
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import train_test_split
from utils import get_tweet_embedding, classify_polarity, map_classes


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


def process_data(df, model, embedding_dim):
    df['polarity'] = df['clean_text'].apply(classify_polarity)
    X = np.vstack(df['clean_text'].apply(lambda x: get_tweet_embedding(x, model, embedding_dim)).values)
    y = df['polarity'].values
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def save_and_upload_data(X_train, X_val, y_train, y_val, bucket_name, prefix='train'):
    hoy = date.today()
    train_data = np.hstack((map_classes(y_train).reshape(-1, 1), X_train))
    val_data = np.hstack((map_classes(y_val).reshape(-1, 1), X_val))

    path_to_train = f"{prefix}_{hoy}.csv"
    path_to_val = f"validation_{hoy}.csv"

    np.savetxt(path_to_train, train_data, delimiter=",", fmt="%.6f")
    np.savetxt(path_to_val, val_data, delimiter=",", fmt="%.6f")

    s3 = boto3.client('s3')
    s3.upload_file(path_to_train, bucket_name, path_to_train)
    s3.upload_file(path_to_val, bucket_name, path_to_val)

    return f's3://{bucket_name}/{path_to_train}', f's3://{bucket_name}/{path_to_val}'


def train_model(s3_train_path, s3_val_path, bucket_name):
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()
    s3_output_path = f"s3://{bucket_name}/output/"

    xgb_estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve('xgboost', sagemaker_session.boto_region_name, version='1.5-1'),
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        output_path=s3_output_path
    )

    xgb_estimator.set_hyperparameters(
        objective='multi:softmax',
        num_round=100,
        max_depth=5,
        eta=0.2,
        num_class=3,
        eval_metric='mlogloss'
    )

    train_input = TrainingInput(s3_data=s3_train_path, content_type='text/csv')
    validation_input = TrainingInput(s3_data=s3_val_path, content_type='text/csv')
    xgb_estimator.fit({'train': train_input, 'validation': validation_input})
    return xgb_estimator


def save_training_metrics(estimator, output_file="evaluation_metrics.json"):
    training_job = estimator.latest_training_job.describe()
    metrics = training_job["FinalMetricDataList"]
    hyperparameters = training_job["HyperParameters"]

    model_metrics = {
        "metrics": {m["MetricName"]: m["Value"] for m in metrics},
        "hyperparameters": hyperparameters,
        "training_job_name": training_job["TrainingJobName"]
    }

    with open(output_file, "w") as f:
        json.dump(model_metrics, f, indent=4)


def main():
    env_vars = load_env_vars()
    query = """
        SELECT *
        FROM public.tweets_process
        WHERE created_at <= '2017-11-30'
        ORDER BY created_at DESC;
    """
    df = get_db_data(env_vars, query)

    embedding_model = download_and_prepare_embedding(
        bucket_name='embedding-tweets',
        file_key='glove.twitter.27B.50d.txt',
        local_file_path='/tmp/glove.twitter.27B.50d.txt'
    )

    X_train, X_val, y_train, y_val = process_data(df, embedding_model, embedding_dim=50)

    s3_train_path, s3_val_path = save_and_upload_data(
        X_train, X_val, y_train, y_val, bucket_name='data-train-tweets-models'
    )

    print(f"Train data en S3: {s3_train_path}")
    print(f"Validation data en S3: {s3_val_path}")

    estimator = train_model(s3_train_path, s3_val_path, bucket_name='data-train-tweets-models')

    save_training_metrics(estimator)


if __name__ == "__main__":
    main()
