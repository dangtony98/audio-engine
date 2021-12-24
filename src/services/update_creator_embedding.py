from app import db
import numpy as np
from bson.objectid import ObjectId
import pickle    
import os 

NUMBER_OF_CREATORS_TO_RECOMMEND = 20
INVERSE_ORDER = -1
OBJECT_ID_INDEX = 0
PROB_INDEX = 1
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


dict_filter = lambda x, y: dict([(i,x[i]) for i in x if i in set(y)])


# def pickle_word_embeddings():
#     """ 
#     USE IF YOU NEED TO PICKLE EMBEDDINGS
#     """
#     index = 0
#     # TODO: fix embeddings_dict not found issue and the path issue
#     with open("src/services/word_embeddings/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
#         for line in f:
#             index += 1
#             print(index)
#             values = line.split()
#             word = values[0]
#             vector = np.asarray(values[1:], "float32")
#             embeddings_dict[word] = vector
#     with open('embeddings.pickle', 'wb') as handle:
#         pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def query_transcriptions(user_id):
    """
    Get all the audios for a certain user where transcription exists and concatenate their transcfriotions 
    """
    audios = db.audios.find({"user": ObjectId(user_id), "transcription": {"$exists":1}})
    audio_transcriptions = ''
    for audio in audios:
        audio_transcriptions += " " + audio['transcription']
    
    return audio_transcriptions


def delete_stopwords(audio_transcriptions):
    """
    Delete stopwords from the audio transcriptions (a, the, and, etc.)
    """
    with open(DIR_PATH + '/word_embeddings/stopwords.pickle', 'rb') as handle:
        stopwords = pickle.load(handle)
    transcriptions_without_stop_words = [word for word in audio_transcriptions.split() if not word in stopwords]
    return transcriptions_without_stop_words


def load_embeddings():
    """
    Load teh previously pickled embeddings
    """
    with open(DIR_PATH + '/word_embeddings/embeddings.pickle', 'rb') as handle:
        embeddings_dict = pickle.load(handle)
    return embeddings_dict


def calculate_embedding(embeddings_dict, transcriptions_without_stop_words):
    """
    Calculate creator embeddings by putting them through word embeddings or making them 0's if those don't exist
    """
    creator_embedding = np.mean([embeddings_dict[audio] for audio in transcriptions_without_stop_words if audio in embeddings_dict], axis=0).tolist()
    if creator_embedding != creator_embedding:
        creator_embedding = [0] * 50
    return creator_embedding


def generate_creator_embedding(user_id):
    embeddings_dict = load_embeddings()

    audio_transcriptions = query_transcriptions(user_id)

    transcriptions_without_stop_words = delete_stopwords(audio_transcriptions)

    creator_embedding = calculate_embedding(embeddings_dict, transcriptions_without_stop_words)

    return creator_embedding


def push_embedding_to_db(user_id, creator_embedding):
    """
    Update the creator embedding of a certain user
    """
    db.users.update_one(
        {"_id": ObjectId(user_id)}, 
        {"$set": {"creatorEmbedding": creator_embedding}},
        upsert=True
    )
