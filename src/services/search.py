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
    Query the id and title for all the audios. Potentially also hashtags
    """
    audios = [[audio["_id"], audio["wordEmbedding"]] for audio in list(db.audios.find({"wordEmbedding": {"$exists":1}}, {"wordEmbedding": 1}))]
    return audios


def load_embeddings_stopwords():
    """
    This function loads the previously pickled word embeddings and stopwords
    """
    with open(DIR_PATH + '/word_embeddings/stopwords.pickle', 'rb') as handle:
        stopwords = pickle.load(handle)
    with open(DIR_PATH + '/word_embeddings/embeddings_twitter.pickle', 'rb') as handle:
        embeddings_dict = pickle.load(handle)

    return embeddings_dict, stopwords


def convert_to_embeddings(title, embeddings_dict, stopwords):
    """
    This function coverts every title (potentially consisting of multiple words) into word embeddings
    It also disregards any stopwords
    """
    title = title.lower()
    new_title = np.mean([embeddings_dict[word] for word in title.split() if (word not in stopwords) and (word in embeddings_dict)], axis=0)
    return new_title


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


def calc_distances(query, audios):
    """
    Calculate the mean cos distances betweed the words in query and the words in 
    """
    # distances = [[audio[0], np.mean([spatial.distance.cosine(query_word, word) for query_word in query for word in audio[1]])] for audio in audios]
    distances = [[audio[0], np.mean([spatial.distance.cosine(query, audio[1])])] for audio in audios]
    return distances


def sort_ascending(distances):
    """
    Sort every audio from the smallest to the largest mean distance
    """
    audio_ids = set(map(lambda x:x[0], distances))
    mean_distances = [[id, np.mean([audio[1] for audio in distances if audio[0]==id])] for id in audio_ids]
    mean_distances = sorted([[id, np.mean([audio[1] for audio in mean_distances if audio[0]==id])] for id in audio_ids], key=itemgetter(1))
    return mean_distances


def get_audio_data(sorted_mean_distances):
    """
    Get the data for selected audios
    """
    ranking = [distance[0] for distance in sorted_mean_distances][0:30]
    search_results = list(db.audios.find({"_id": {"$in": ranking}}))
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
    # Query the data and load everything 
    audios = get_start_data()
    # embeddings_dict, stopwords = load_embeddings_stopwords()

    # Convert audios and the query to word embeddings amd remove stopwords
    # audios = [[audio[0], convert_to_embeddings(audio[1], embeddings_dict, stopwords)] for audio in audios]
    query = convert_to_embeddings(query, embeddings_dict, stopwords)

    # Calculate mean cosine distanecs between each word of the query and each word in the titles of audios
    mean_distances = calc_distances(query, audios)
    print(mean_distances)
    sorted_mean_distances = sort_ascending(mean_distances)[0:NUMBER_OF_SEARCH_RESULTS]
    print(sorted_mean_distances)

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
