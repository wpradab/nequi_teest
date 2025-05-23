import re
import numpy as np
from textblob import TextBlob
from langdetect import detect, LangDetectException

def safe_detect_language(text):
    if isinstance(text, str) and text.strip():
        try:
            return detect(text)
        except LangDetectException:
            # Catch any LangDetectException and return 'unknown'
            return 'unknown'
    else:
        return 'unknown'


def preprocess_tweet(text, remove_stopwords=True):
    # 1. Minúsculas
    text = text.lower()

    # 2. Eliminar menciones
    text = re.sub(r'@\w+', '', text)

    # 3. Eliminar URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # 4. Eliminar hashtags (solo el símbolo #)
    text = text.replace("#", "")

    # 5. Eliminar emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # 6. Eliminar signos de puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 7. Eliminar números
    text = re.sub(r'\d+', '', text)

    # 9. Eliminar espacios extra
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_tweet_embedding(text, embedding_model, embedding_dim=100):
    words = text.split()
    word_vectors = []
    for word in words:
        if word in embedding_model:
            word_vectors.append(embedding_model[word])
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(embedding_dim)


# Función para obtener polaridad
def classify_polarity(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 1
    elif polarity < -0.1:
        return -1
    else:
        return 0

# 📌 Mapear clases
def map_classes(y):
    return np.where(y == -1, 0, np.where(y == 0, 1, 2))

def inverse_map_classes(y_mapped):
    return np.where(y_mapped == 0, -1, np.where(y_mapped == 1, 0, 1))