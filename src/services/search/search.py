from app import db
from datetime import date
import numpy as np
from bson.objectid import ObjectId
from scipy import spatial
from operator import itemgetter
from datetime import datetime, timedelta
import pickle    
import os 
from tasks import embeddings_dict, stopwords

NUMBER_OF_SEARCH_RESULTS = 30
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


dict_filter = lambda x, y: dict([(i,x[i]) for i in x if i in set(y)])


def get_start_data():
    """
    Query the ids and wordEmbeedings for all the audios
    """
    print("Start1")
    audios = [[audio["_id"], audio["wordEmbedding"]] for audio in list(db.audios.find({"wordEmbedding": {"$exists":1}}, {"wordEmbedding": 1}))]
    print("Finish1")
    return audios


def load_embeddings_stopwords():
    """
    This function loads the previously pickled word embeddings and stopwords
    """
    with open(DIR_PATH + '/../word_embeddings/stopwords.pickle', 'rb') as handle:
        stopwords = pickle.load(handle)
    with open(DIR_PATH + '/../word_embeddings/embeddings_twitter.pickle', 'rb') as handle:
        embeddings_dict = pickle.load(handle)

    return embeddings_dict, stopwords


def convert_to_embedding(query):
    """
    This function coverts the query (potentially consisting of multiple words) into word embeddings, and takes the mean of those
    It also disregards any stopwords, and converts teh query to lowercase characters
    """
    embeddings_query = np.mean([embeddings_dict[word] for word in query.lower().split() if (word not in stopwords) and (word in embeddings_dict)], axis=0)
    return embeddings_query


def clean_search_results(item):
    """
    Convert ObjectIds to Strings in order to jsonify afterwards
    """
    item["_id"] = str(item["_id"])
    try:
        item["user"] = str(item["user"])
    except KeyError:
        print("The audio" + item["_id"]  + "has no attribute user")
    try:
        item["rss"] = str(item["rss"])
    except KeyError:
        print("The audio" + item["_id"]  + "has no attribute rss")
    return item


def calculate_distances(query, audios):
    """
    Calculate the mean cos distances betweed the query and the audio embeddings
    """
    # distances = [[audio[0], np.mean([spatial.distance.cosine(query_word, word) for query_word in query for word in audio[1]])] for audio in audios]
    distances = [[audio[0], spatial.distance.cosine(query, audio[1])] for audio in audios]
    return distances


def sort_ascending(distances):
    """
    Sort every audio from the smallest to the largest mean distance
    """
    # Getting the unique set of audio_ids that we need to rank
    mean_distances = sorted(distances, key=itemgetter(1))
    return mean_distances


def get_audio_data(sorted_mean_distances):
    """
    Get the data for selected audios
    """
    ranking = [distance[0] for distance in sorted_mean_distances]
    search_results = list(db.audios.find({"_id": {"$in": ranking}}, 
                        {"title": 1, "url": 1, "user": 1, "duration":1, "rss": 1, "listens": 1}))
    return search_results, ranking


def order_search_results(search_results, ranking):
    """
    Order it according to the ranking
    """
    order_dict = {id: index for index, id in enumerate(ranking)}
    sorted_search_results = sorted(search_results, key=lambda x: order_dict[x["_id"]])
    sorted_search_results = [clean_search_results(result) for result in sorted_search_results]
    return sorted_search_results


def get_search_audio_resuls(query):
    # Query the data consisting of potential 
    audios = get_start_data()

    # Convert audios and the query to word embeddings amd remove stopwords
    # audios = [[audio[0], convert_to_embeddings(audio[1], embeddings_dict, stopwords)] for audio in audios]
    query = convert_to_embedding(query)

    # Calculate mean cosine distanecs between each word of the query and each word in the titles of audios
    mean_distances = calculate_distances(query, audios)
    sorted_mean_distances = sort_ascending(mean_distances)[0:NUMBER_OF_SEARCH_RESULTS]
    print(len(sorted_mean_distances))

    # Get the data for the necessary audios and sort it in the right way
    search_results, ranking = get_audio_data(sorted_mean_distances)
    sorted_search_results = order_search_results(search_results, ranking)
    print([result["title"] for result in sorted_search_results])

    return sorted_search_results


def push_search_to_db(user_id, search_query, search_results):
    values = {}
    values["user"] = ObjectId(user_id)
    values["createdAt"] = datetime.now() 
    values["query"] = search_query
    values["results"] = [item["_id"] for item in search_results]

    twenty_seconds_ago = datetime.now() - timedelta(seconds=15)
    new_values = {"$set": values}

    db.searches.update_one({"user": ObjectId(values["user"]), "createdAt": {"$gte": twenty_seconds_ago}}, new_values, upsert=True)
