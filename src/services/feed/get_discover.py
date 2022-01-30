from bson.objectid import ObjectId
import numpy as np
from cvxpy import *
from app import db
from datetime import datetime, timezone
import os
from app import r
from ..utils import calculate_embedding

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
INVERSE_ORDER = -1
PREFERENCES = ['entertainment', 'comedy', 'daily life', 'storytelling', 'arts', 'music', 'fashion beauty',
            'health fitness sport', 'sports', 'diy', 'true crime', 'fiction', 'dating', 'parenting', 'food', 'travel', 
            'languages', 'nature', 'history', 'religion', 'society', 'culture', 'education', 'science', 'career', 'business', 
            'tech', 'finance investing', 'politics', 'news']
OPTION = "AVG"


def get_user_x(user_id):
    """
    gets embedding for user with id [user_id]
    """
    return list(db.users.find({"_id": ObjectId(user_id)}))[0]["initEmbedding"]


def get_content_pool(user_id, redis_ids):
    """
    Returns a content pool of ids to consider and their repective embeddings
    """
    # get ids of seen ledges
    seen_ids = [l["audio"] for l in db.ratings.find({"user": ObjectId(user_id)})]

    # get ids of blocked users
    blocked_users = [b["to"] for b in db.blocks.find({"from": ObjectId(user_id)})]

    # get ids of blocked content
    blocked_ids = [a["_id"] for a in db.audios.find({"user": {"$in": blocked_users}}, {"_id": 1})]

    unseen_pool = list(
        db.audios.find({"_id": {"$nin": list(set(seen_ids) | set(blocked_ids) | set(redis_ids))}, "isVisible": True, "wordEmbedding": {"$exists": 1}}, 
        {"wordEmbedding": 1})
    )
    see_ids = [str(id) for id in seen_ids]
    return unseen_pool, seen_ids


def get_sorted_content(mongo_scores, unseen_redis_scores):
    """
    Return sorted content ids
    """

    scores = dict(mongo_scores, **unseen_redis_scores)
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: -item[1])[:100]}
    sorted_scores_keys = [ObjectId(id) for id in sorted_scores.keys()]
    return sorted_scores_keys


def get_redis_scores(user_id):
    """
    Gets all the cached scores (corresponding to audios) for a certain user
    """
    try:
        redis_scores = r.hgetall(user_id)
    except Exception as e: 
        redis_scores = {}
        print(e)
    redis_ids = redis_scores.keys()
    return redis_scores, redis_ids


def get_user_preference_vector(user_id):
    """
    Gets user's init embedding and converts it to the the average of word embeddings
    """
    user_x = get_user_x(user_id)
    user_preferences = [PREFERENCES[i] for i in range(len(PREFERENCES)) if user_x[i]==1]
    if OPTION == "AVG":
        user_preferences = np.mean([calculate_embedding(user_preference) for user_preference in user_preferences], axis=0)
    elif OPTION == "MAX": 
        user_preferences = [calculate_embedding(user_preference) for user_preference in user_preferences]
    return user_preferences


def send_to_redis(user_id, mongo_scores):
    """
    Sends the new scores to redis if there is anything to send
    """
    try: 
        if len(mongo_scores) > 0:
            r.hmset(user_id, mongo_scores)
    except Exception as e: 
        print(e)


def get_data(sorted_scores_keys):
    """
    Gets the data from Mongo for selected audios
    """
    feed = list(
        db.audios.find({"_id": {"$in": sorted_scores_keys}}, 
        {"title": 1, "url": 1, "user": 1, "duration":1, "listens": 1})
    )
    return feed


def get_feed(user_id):
    """
    gets the discovery feed for user with id [user_id]
    """
    # Get user's pereferences
    user_preferences = get_user_preference_vector(user_id)

    # Get the data cached in Redis; ids = ids of cached audios; scores = ids & scores
    redis_scores, redis_ids = get_redis_scores(user_id)
    
    # Query from DB everything besides the ids cached in Redis
    unseen_pool, seen_ids = get_content_pool(user_id, [ObjectId(id) for id in redis_ids])

    # Filter only unseen scores from redis; and unseen word Embedidngs from mongo; calculate scores for mongo audios
    unseen_redis_scores = {key: float(redis_scores[key]) for key in redis_ids if key not in seen_ids}
    unseen_embeddings = {audio["_id"]: audio["wordEmbedding"] for audio in unseen_pool}
    if OPTION == "AVG":
        mongo_scores = {str(episode_id): np.dot(user_preferences, episode_embedding) for episode_id, episode_embedding in unseen_embeddings.items()}
    elif OPTION == "MAX":
        mongo_scores = {str(episode_id): np.max([np.dot(user_preference, episode_embedding) for user_preference in user_preferences]) for episode_id, episode_embedding in unseen_embeddings.items()}

    # Send the new scores to redis
    send_to_redis(user_id, mongo_scores)

    # Rank audios and take the first 100
    sorted_scores_keys = get_sorted_content(mongo_scores, unseen_redis_scores)

    # Query the data for those 100 audios from Mongo
    feed = get_data(sorted_scores_keys)

    # Sort data in the same order
    order_dict = {_id: index for index, _id in enumerate(sorted_scores_keys)}
    feed.sort(key=lambda x: order_dict[x["_id"]])
    
    # TODO: Add user if I want to have more diversity
    # creators = {audio["user"]: 0 for audio in sorted_unseen_pool}
    # def diversity_threshold_check(user_id):
    #     creators[user_id] += 1
    #     if creators[user_id] <= 12: 
    #         return True
    #     return False
    # feed = [audio for audio in sorted_unseen_pool if diversity_threshold_check(audio["user"])][:100]

    return feed


def update_feed(user_id, feed_name, feed):
    """
    update feed named [feed_name] for user with id [user_id]
    """

    # filter = {"user": ObjectId(user_id)}

    values = {}
    values[feed_name] = [item["_id"] for item in feed]
    values["user"] = ObjectId(user_id)
    values["createdAt"] = datetime.now(timezone.utc)
    # new_values = {"$set": values}

    # db.feeds.update_one(filter, new_values, upsert=True)
    db.feeds.insert_one(values)
