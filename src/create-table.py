import os
import psycopg2
from dotenv import load_dotenv

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

# Query para crear la tabla
create_table_query = """
DROP TABLE IF EXISTS public.tweets_process;

CREATE TABLE public.tweets_process (
    tweet_id bigint ENCODE az64,
    author_id character varying(256) ENCODE lzo,
    inbound boolean,
    created_at timestamp,
    text character varying(6000),
    response_tweet_id character varying(15000),
    in_response_to_tweet_id character varying(256),
    clean_text character varying(6000)
) DISTSTYLE AUTO;
"""

# Ejecutar la query
cur.execute(create_table_query)

# Confirmar cambios
conn.commit()

# Cerrar conexiones
cur.close()
conn.close()

print("Tabla public.tweets creada exitosamente.")
