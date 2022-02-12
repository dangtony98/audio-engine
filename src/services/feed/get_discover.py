from bson.objectid import ObjectId
from collections import Counter
import numpy as np
from cvxpy import *
from app import db
from datetime import datetime, timezone, date
import os
from app import r
from operator import itemgetter
from ..utils import calculate_embedding

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
INVERSE_ORDER = -1
ANNOYANCE_THRESHOLD = 2
TOP_FEED_THRESHOLD = 4
REDIS_THRESHOLD = 1000 # How many items do we wanna save in redis
FOLLOWING_BENEFIT = 0.15
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


def get_content_pool(user_id, redis_ids, redis_last_date):
    """
    Returns a content pool of ids to consider and their repective embeddings
    """
    # get ids of seen ledges
    listened_ids = [l["audio"] for l in db.ratings.find({"user": ObjectId(user_id)})]

    # get ids of blocked users
    blocked_users = [b["to"] for b in db.blocks.find({"from": ObjectId(user_id)})]

    # get ids of blocked content
    blocked_ids = [a["_id"] for a in db.audios.find({"user": {"$in": blocked_users}}, {"_id": 1})]

    redis_last_date = '2022-01-01 00:00:00.000+05:00' if redis_last_date == None else redis_last_date
    nonlistened_pool = list(
        db.audios.find({"_id": {"$nin": list(set(listened_ids) | set(blocked_ids) | set(redis_ids))}, 
                        "isVisible": True, 
                        "wordEmbedding": {"$exists": 1},
                        "createdAt": {"$gte": datetime.fromisoformat(str(redis_last_date))}}, 
        {"wordEmbedding": 1})
    )
    listened_ids = [str(id) for id in listened_ids]
    return nonlistened_pool, listened_ids


def add_following_benefit(scores, user_id):
    """
    Seeing which audios come from people the user is following and increasing their ranking position
    """
    audio_creators = list(db.audios.find({"_id": {"$in": [ObjectId(score) for score in scores.keys()]}}, {"user": 1}))
    audio_creators_set = list(set([creator["user"] for creator in audio_creators]))
    follows_set = set([follow["to"] for follow in db.follows.find({"from": ObjectId(user_id), "to": {"$in": audio_creators_set}}, {"to": 1})])
    audios_from_follows = [str(audio["_id"]) for audio in audio_creators if audio["user"] in follows_set]
    scores = {k: (v + (FOLLOWING_BENEFIT if k in audios_from_follows else 0)) for k, v in scores.items()}
    return scores


def get_sorted_content(mongo_scores, unseen_redis_scores, user_id, seen_redis_scores):
    """
    Return sorted content ids
    Delete listened/anniying ids from redis and anything that goes above 1000 audio ids
    """
    scores = dict(mongo_scores, **unseen_redis_scores)
    try:
        r.hdel("user:" + user_id + ":scores", *seen_redis_scores.keys())
        scores_to_be_deleted = dict(sorted(scores.items(), key = itemgetter(1), reverse = True)[1000:])
        r.hdel("user:" + user_id + ":scores", *scores_to_be_deleted.keys())
    except Exception as e:
        print(e)
    
    # This block of code actually benefits the audios from the account a  user is following
    scores = add_following_benefit(scores, user_id)

    sorted_scores = {k: v + np.random.normal(0, 0.5) for k, v in sorted(scores.items(), key=lambda item: -item[1])[:150]}
    sorted_scores = {k: v for k, v in sorted(sorted_scores.items(), key=lambda item: -item[1])[:100]}
    sorted_scores_keys = [ObjectId(id) for id in sorted_scores.keys()]
    return sorted_scores_keys


def get_redis_scores(user_id):
    """
    Gets all the cached scores (corresponding to audios) for a certain user and the timestamp when those scores were generate
    """
    try:
        redis_scores = r.hgetall("user:" + user_id + ":scores")
    except Exception as e: 
        redis_scores = {}
        print(e)
    try:
        redis_last_date = r.get("user:" + user_id + ":lastUpdateDate")
    except Exception as e: 
        redis_last_date = '2022-01-01 00:00:00.000+05:00'
        print(e)
    redis_ids = redis_scores.keys()
    return redis_scores, redis_ids, redis_last_date


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
    # case: user has chosen no preferences during onboarding
    if (user_preferences != user_preferences).all():
        user_preferences = [0] * 25
    return user_preferences


def send_to_redis(user_id, mongo_scores):
    """
    Sends the new scores to redis if there is anything to send
    Also send the current timestamp to know when these scores were updated
    """
    try: 
        if len(mongo_scores) > 0:
            r.hmset("user:" + user_id + ":scores", mongo_scores)
        r.set("user:" + user_id + ":lastUpdateDate", datetime.now().isoformat())
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


def filter_annoying_audios(user_id):
    """
    This function filter out audios tha toccur too often on the top of the feed
    """
    seen_feeds = [feed["discover"][:TOP_FEED_THRESHOLD] for feed in db.feeds.find({"user": ObjectId(user_id)})]
    annoying_audio_ids = [str(audio_id) for feed in seen_feeds for audio_id in feed]
    cnt = Counter(annoying_audio_ids)
    annoying_audio_ids = [audio_id for audio_id, occurences in cnt.items() if occurences >= ANNOYANCE_THRESHOLD]
    return annoying_audio_ids


def get_feed(user_id):
    """
    gets the discovery feed for user with id [user_id]
    """
    # Get user's pereferences
    user_preferences = get_user_preference_vector(user_id)

    # Get the data cached in Redis; ids = ids of cached audios; scores = ids & scores
    redis_scores, redis_ids, redis_last_date = get_redis_scores(user_id)
    
    # Query from DB everything besides the ids cached in Redis
    nonlistened_pool, listened_ids = get_content_pool(user_id, [ObjectId(id) for id in redis_ids], redis_last_date)

    # Filter out the items that were previously seen too many times on the top of the feed - annoying
    annoying_audio_ids = filter_annoying_audios(user_id)

    # Filter only unseen scores from redis; and unseen word Embedidngs from mongo; calculate scores for mongo audios
    nonlistened_redis_scores = {key: float(redis_scores[key]) for key in redis_ids if (key not in listened_ids) and (key not in annoying_audio_ids)}
    listened_redis_scores = {key: float(redis_scores[key]) for key in redis_ids if (key in listened_ids) or (key in annoying_audio_ids)}
    nonlistened_audios_embeddings = {audio["_id"]: audio["wordEmbedding"] for audio in nonlistened_pool if audio["_id"] not in annoying_audio_ids}
    if OPTION == "AVG":
        mongo_scores = {str(episode_id): np.dot(user_preferences, episode_embedding) for episode_id, episode_embedding in nonlistened_audios_embeddings.items()}
    elif OPTION == "MAX":
        mongo_scores = {str(episode_id): np.max([np.dot(user_preference, episode_embedding) for user_preference in user_preferences]) for episode_id, episode_embedding in nonlistened_audios_embeddings.items()}

    # Get 1000 best audios
    mongo_scores = dict(sorted(mongo_scores.items(), key = itemgetter(1), reverse = True)[:REDIS_THRESHOLD])

    # Send the new scores to redis
    send_to_redis(user_id, mongo_scores)

    # Rank audios and take the first 100
    sorted_scores_keys = get_sorted_content(mongo_scores, nonlistened_redis_scores, user_id, listened_redis_scores)

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
