import os
import json
import boto3
import botocore
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


hoy = date.today()

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
WHERE created_at <= '2017-11-30'
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

df['polarity'] = df['clean_text'].apply(classify_polarity)

X = np.vstack(df['clean_text'].apply(lambda x: get_tweet_embedding(x, model, embedding_dim)).values)
y = df['polarity'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Mapear las clases y preparar los datos para CSV
train_data = np.hstack((map_classes(y_train).reshape(-1, 1), X_train))
val_data = np.hstack((map_classes(y_val).reshape(-1, 1), X_val))

# Guardar CSVs sin cabecera ni índice
path_to_train_data = f"train_{hoy}.csv"
path_to_val_data = f"validation_{hoy}.csv"

np.savetxt(path_to_train_data, train_data, delimiter=",", fmt="%.6f")
np.savetxt(path_to_val_data, val_data, delimiter=",", fmt="%.6f")

# Subir a S3
s3 = boto3.client('s3')
bucket_name = 'data-train-tweets-models'

s3.upload_file(path_to_train_data, bucket_name, path_to_train_data)
s3.upload_file(path_to_val_data, bucket_name, path_to_val_data)

# Rutas S3 para SageMaker
s3_train_path = f's3://{bucket_name}/{path_to_train_data}'
s3_val_path = f's3://{bucket_name}/{path_to_val_data}'

print(f'Train data en S3: {s3_train_path}')
print(f'Validation data en S3: {s3_val_path}')

# Configurar estimator (igual que antes)
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

# Inputs para entrenamiento y validación
train_input = TrainingInput(s3_data=s3_train_path, content_type='text/csv')
validation_input = TrainingInput(s3_data=s3_val_path, content_type='text/csv')

# Lanzar training job con validación
xgb_estimator.fit({'train': train_input, 'validation': validation_input})

# Obtener métricas del job de entrenamiento
training_job = xgb_estimator.latest_training_job.describe()
metrics = training_job["FinalMetricDataList"]
hyperparameters = training_job["HyperParameters"]

# Formatear métricas como diccionario
model_metrics = {
    "metrics": {m["MetricName"]: m["Value"] for m in metrics},
    "hyperparameters": hyperparameters,
    "training_job_name": training_job["TrainingJobName"]
}

# Guardar archivo JSON localmente
with open("evaluation_metrics.json", "w") as f:
    json.dump(model_metrics, f, indent=4)

# Subir a S3
s3 = boto3.client('s3')
s3.upload_file("evaluation_metrics.json", "data-train-tweets-models", "evaluation_metrics.json")

sm_client = boto3.client('sagemaker')

response = sm_client.list_training_jobs(MaxResults=5)

job_name = response['TrainingJobSummaries'][0]['TrainingJobName']

response = sm_client.describe_training_job(TrainingJobName=job_name)

model_artifact = response['ModelArtifacts']['S3ModelArtifacts']
role_arn = response['RoleArn']
image_uri = response['AlgorithmSpecification']['TrainingImage']

print("Model Artifact S3 Path:", model_artifact)
print("Execution Role:", role_arn)
print("Image URI:", image_uri)

model_name = job_name + "-model"

create_model_response = sm_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role_arn,
    PrimaryContainer={
        'Image': image_uri,
        'ModelDataUrl': model_artifact
    }
)

print("Modelo creado:", create_model_response['ModelArn'])

model_package_group_name = 'xgboost-registry'  # Puedes usar un nombre propio

# Crear (una vez) el grupo si no existe
try:
    sm_client.create_model_package_group(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageGroupDescription='Registro de modelos XGBoost'
    )
    print("Model Package Group creado.")
except botocore.exceptions.ClientError as error:
    if error.response['Error']['Code'] == 'ValidationException' and 'already exists' in error.response['Error']['Message']:
        print("Model Package Group ya existe.")
    else:
        raise  # Lanza cualquier otro error que no sea este
# Registrar el modelo
model_package_response = sm_client.create_model_package(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageDescription='Versión registrada del modelo XGBoost',
    InferenceSpecification={
        'Containers': [
            {
                'Image': image_uri,
                'ModelDataUrl': model_artifact
            }
        ],
        'SupportedContentTypes': ['text/csv'],
        'SupportedResponseMIMETypes': ['text/csv']
    },
    ModelMetrics={
        'ModelQuality': {
            'Statistics': {
                'ContentType': 'application/json',
                'S3Uri': "s3://data-train-tweets-models/evaluation_metrics.json"
            }
        }
    },
    ModelApprovalStatus='PendingManualApproval'
)

print("Modelo registrado con ARN:", model_package_response['ModelPackageArn'])


sm_client = boto3.client('sagemaker')

sm_client.update_model_package(
    ModelPackageArn=model_package_response['ModelPackageArn'],
    ModelApprovalStatus='Approved'
)
