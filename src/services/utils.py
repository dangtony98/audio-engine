from datetime import date
import numpy as np
import os
import pickle

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
EMBEDDING_SIZE = 25


def compute_age(birthdate):
    """
    Compute age given the person's birthdate
    """
    try:
        today = date.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    except:
        age = 0
    return age


def load_preference_embeddings():
    """
    Load the previously pickled embeddings
    """
    with open(DIR_PATH + '/word_embeddings/preference_embeddings_twitter.pickle', 'rb') as handle:
        embeddings_dict = pickle.load(handle)
    return embeddings_dict


def calculate_embedding(words):
    """
    Calculate creator embeddings by putting them through word embeddings or making them 0's if those don't exist (taking a mean)
    """
    embeddings_dict = load_preference_embeddings()
    creator_embedding = np.mean([embeddings_dict[word] for word in words.split() if word in embeddings_dict], axis=0).tolist()
    if creator_embedding != creator_embedding:
        creator_embedding = [0] * EMBEDDING_SIZE
    return creator_embedding


def clean_output(pool):
    """
    clean feed to be compatible with Flask's jsonify
    """

    def clean(item):
        item["_id"] = str(item["_id"])
        item["user"] = str(item["user"])
        return item

    return [clean(item) for item in pool]



# # NOT USED:

# def delete_stopwords(audio_transcriptions):
#     """
#     Delete stopwords from the audio transcriptions (a, the, and, etc.)
#     """
#     with open(DIR_PATH + '/../word_embeddings/stopwords.pickle', 'rb') as handle:
#         stopwords = pickle.load(handle)
#     transcriptions_without_stop_words = " ".join([word for word in audio_transcriptions.split() if not word in stopwords])
#     return transcriptions_without_stop_words