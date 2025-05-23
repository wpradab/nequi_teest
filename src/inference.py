import os
import time
import boto3
import psycopg2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gensim.models import KeyedVectors
from utils import get_tweet_embedding, inverse_map_classes


# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Leer variables desde el entorno
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
dbname = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')

# Usar las variables en la conexión
conn = psycopg2.connect(
    host=host,
    port=port,
    dbname=dbname,
    user=user,
    password=password
)

cur = conn.cursor()

# Leer la tabla como DataFrame
query = """
SELECT *
FROM public.tweets_process
WHERE created_at >= '2017-11-30'
ORDER BY created_at DESC
;
"""

df = pd.read_sql(query, conn)

# Cerrar conexión
conn.close()

s3 = boto3.client('s3')

# Parámetros
bucket_name = 'embedding-tweets'
file_key = 'glove.twitter.27B.50d.txt'
local_file_path = '/tmp/glove.twitter.27B.50d.txt'

# Descargar archivo desde S3
s3.download_file(bucket_name, file_key, local_file_path)

# Contar número de líneas
num_lines = sum(1 for line in open(local_file_path))
embedding_dim = 50

# Crear un archivo nuevo con cabecera
with open(local_file_path, 'r') as original, open('/tmp/glove_fixed.txt', 'w') as fixed:
    fixed.write(f"{num_lines} {embedding_dim}\n")
    for line in original:
        fixed.write(line)

model = KeyedVectors.load_word2vec_format('/tmp/glove_fixed.txt', binary=False)

X = np.vstack(df['clean_text'].apply(lambda x: get_tweet_embedding(x, model, embedding_dim)).values)
payload = "\n".join(",".join(f"{v:.2f}" for v in row) for row in X)

sm_client = boto3.client('sagemaker')
response = sm_client.list_training_jobs(MaxResults=5)
job_name = response['TrainingJobSummaries'][0]['TrainingJobName']

response = sm_client.describe_training_job(TrainingJobName=job_name)

model_artifact = response['ModelArtifacts']['S3ModelArtifacts']
role_arn = response['RoleArn']
image_uri = response['AlgorithmSpecification']['TrainingImage']

timestamp = str(int(time.time()))
model_name = f"xgboost-model-from-package-{timestamp}"
endpoint_name = f"xgboost-registry-endpoint-{timestamp}"

create_model_response = sm_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role_arn,  # Ya lo tienes del training job
    PrimaryContainer={
        'Image': image_uri,
        'ModelDataUrl': model_artifact  # Este es el S3 del modelo
    }
)

print("Modelo creado:", create_model_response['ModelArn'])
sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_name + "-config",
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,  # ¡Aquí va ModelName, no ModelPackageName!
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large'
        }
    ]
)

sm_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_name + "-config"
)

import time

while True:
    response = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    print("Estado del endpoint:", status)
    if status in ['InService', 'Failed']:
        break
    time.sleep(15)


runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='text/csv',
    Body=payload
)
result = response['Body'].read().decode('utf-8')

y_pred = [float(x) for x in result.strip().split('\n')]

result_output = pd.Series(y_pred).apply(inverse_map_classes)