import os
import boto3
import socket
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from utils import preprocess_tweet, safe_detect_language

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Leer variables desde el entorno
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
dbname = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')

# Crear cliente de S3
s3 = boto3.client('s3')

# Nombre del bucket
bucket_name = 'tweets-test-123'

# Listar objetos en el bucket
response = s3.list_objects_v2(Bucket=bucket_name)

# Verificar si hay objetos en el bucket
if 'Contents' in response:
    # Ordenar por la fecha de última modificación
    latest_file = max(response['Contents'], key=lambda x: x['LastModified'])

# Leer directamente con pandas
df = pd.read_csv(latest_file, sep='|', nrows= 5000)

df['clean_text'] = df['text'].apply(preprocess_tweet)
df['language'] = df['clean_text'].apply(safe_detect_language)
df_english = df[df['language']=='en']

# df_english['inbound'] = df_english['inbound'].astype(str)
df_english['created_at'] = pd.to_datetime(df_english['created_at'], errors='coerce')

df['clean_text'] = df['text'].apply(preprocess_tweet)
df['language'] = df['clean_text'].apply(safe_detect_language)
df_english = df[df['language']=='en']

# df_english['inbound'] = df_english['inbound'].astype(str)
df_english['created_at'] = pd.to_datetime(df_english['created_at'], errors='coerce')

# Datos de tu cluster Redshift
hostname = 'tweets-teams3.762030194561.us-east-1.redshift-serverless.amazonaws.com'
port = 5439

# Intentar conexión
try:
    sock = socket.create_connection((hostname, port), timeout=10)
    print(f'Conexión exitosa a {hostname}:{port}')
    sock.close()
except Exception as e:
    print(f'Error de conexión: {e}')


# Usar las variables en la conexión
conn = psycopg2.connect(
    host=host,
    port=port,
    dbname=dbname,
    user=user,
    password=password
)

cur = conn.cursor()

for index, row in df_english.iterrows():
    cur.execute("""
        INSERT INTO public.tweets_process (tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id, clean_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (row['tweet_id'], row['author_id'], row['inbound'], row['created_at'], row['text'], row['response_tweet_id'], row['in_response_to_tweet_id'], row['clean_text']))

conn.commit()
cur.close()
conn.close()

