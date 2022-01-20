from ..variables import PREF_TO_INDEX
from bson.objectid import ObjectId
import numpy as np
from cvxpy import *
from app import db
from datetime import datetime
import pickle
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
INVERSE_ORDER = -1


def get_user_x(user_id):
    """
    gets embedding for user with id [user_id]
    """
    return list(db.users.find({"_id": ObjectId(user_id)}))[0]["initEmbedding"]


def get_content_pool(user_id):
    """
    Returns a content pool of ids to consider and their repective embeddings
    """

    # get ids of seen ledges
    seen_ids = [l["audio"] for l in db.listens.find({"user": ObjectId(user_id), "wordEmbedding": {"$exists":1}})]

    # get ids of blocked users
    blocked_users = [b["to"] for b in db.blocks.find({"from": ObjectId(user_id)})]
    # get ids of blocked content
    blocked_ids = [a["_id"] for a in db.audios.find({"user": {"$in": blocked_users}})]

    unseen_pool = list(
        db.audios.find({"_id": {"$nin": list(set(seen_ids) | set(blocked_ids))}, "wordEmbedding": {"$exists":1}})
    )
    seen_pool = list(db.audios.find({"_id": {"$in": seen_ids, "$nin": blocked_ids}}))

    unseen_pool_xs = [a["embedding"] for a in unseen_pool]
    seen_pool_xs = [a["embedding"] for a in seen_pool]

    return seen_pool, seen_pool_xs, unseen_pool, unseen_pool_xs


def get_sorted_content(user_x, content_pool, ratings):
    """
    Return sorted content ratings
    """

    if not content_pool:
        # case: [content_pool] is empty
        return content_pool

    # ratings = np.dot(np.array(content_pool_xs), np.array(user_x))
    sort_idx = ratings.argsort()

    return list(np.array(content_pool)[sort_idx[::-1]])


def get_feed(user_id):
    """
    gets the discovery feed for user with id [user_id]
    """

    PREFERENCES = ['entertainment', 'comedy', 'daily life', 'storytelling', 'arts', 'music', 'fashion beauty',
        'health fitness', 'sports', 'do it yourself', 'true crime', 'dating', 'parenting', 'food', 'travel', 
        'languages', 'history', 'religion', 'society', 'culture', 'education', 'science', 'career', 'business', 
        'tech', 'finance investing', 'politics', 'news']
    user_x = get_user_x(user_id)
    user_preferences = [PREFERENCES[i] for i in range(len(PREFERENCES)) if user_x[i]==1]
    embeddings_dict = load_embeddings()
    user_preferences = np.array([calculate_embedding(embeddings_dict, user_preference) for user_preference in user_preferences])

    seen_pool, seen_pool_xs, unseen_pool, unseen_pool_xs = get_content_pool(user_id)

    # unseen_titles = [audio["title"] for audio in unseen_pool]
    # unseen_titles = [title.lower() for title in unseen_titles]
    # unseen_titles = [delete_stopwords(title) for title in unseen_titles]
    # unseen_titles = np.array([calculate_embedding(embeddings_dict, unseen_title) for unseen_title in unseen_titles])
    unseen_titles = np.array([audio["wordEmbedding"] for audio in unseen_pool])

    mm = []
    for preference in user_preferences:
        mm.append([np.dot(preference, title) for title in unseen_titles])
    dot_prod = np.mean(mm, axis=0)
    dtype = [('_id', ObjectId), ('prob_to_listen', float)]
    result = np.array(list(zip(np.array([audio["_id"] for audio in unseen_pool]), dot_prod)), dtype=dtype)
    result = np.sort(result, order='prob_to_listen')[::INVERSE_ORDER]
    print(result)

    # sorted_seen_pool = get_sorted_content(user_x, seen_pool, seen_pool_xs)
    sorted_unseen_pool = get_sorted_content(user_x, unseen_pool, dot_prod)

    feed = sorted_unseen_pool
    
    return feed


def delete_stopwords(audio_transcriptions):
    """
    Delete stopwords from the audio transcriptions (a, the, and, etc.)
    """
    with open(DIR_PATH + '/word_embeddings/stopwords.pickle', 'rb') as handle:
        stopwords = pickle.load(handle)
    transcriptions_without_stop_words = " ".join([word for word in audio_transcriptions.split() if not word in stopwords])
    return transcriptions_without_stop_words


def load_embeddings():
    """
    Load teh previously pickled embeddings
    """
    with open(DIR_PATH + '/word_embeddings/preference_embeddings_twitter.pickle', 'rb') as handle:
        embeddings_dict = pickle.load(handle)
    return embeddings_dict


def calculate_embedding(embeddings_dict, words):
    """
    Calculate creator embeddings by putting them through word embeddings or making them 0's if those don't exist
    """
    creator_embedding = np.mean([embeddings_dict[word] for word in words.split() if word in embeddings_dict], axis=0).tolist()
    if creator_embedding != creator_embedding:
        creator_embedding = [0] * 25
    return creator_embedding


def update_feed(user_id, feed_name, feed):
    """
    update feed named [feed_name] for user with id [user_id]
    """

    filter = {"user": ObjectId(user_id)}

    values = {}
    values[feed_name] = [item["_id"] for item in feed]
    values["user"] = ObjectId(user_id)
    values["createdAt"] = datetime.now()
    # new_values = {"$set": values}

    # db.feeds.update_one(filter, new_values, upsert=True)
    db.feeds.insert_one(values)


def clean_output(pool):
    """
    clean [pool] to be compatible with Flask's jsonify
    """

    def clean(item):
        item["_id"] = str(item["_id"])
        item["user"] = str(item["user"])
        try:
            item["rss"] = str(item["rss"])
        except KeyError:
            print("The user" + item["_id"]  + "has no attribute rss")
        return item

    return [clean(item) for item in pool]
