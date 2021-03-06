from bson.objectid import ObjectId
from collections import Counter
import numpy as np
from cvxpy import *
from app import db
from datetime import datetime, timezone, timedelta
import os
from app import r
from operator import itemgetter
from ..utils import calculate_embedding


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
INVERSE_ORDER = -1
ANNOYANCE_THRESHOLD = 3
TOP_FEED_THRESHOLD = 4
LAST_FEED_THRESHOLD = 15
REDIS_THRESHOLD = 1000 # How many items do we wanna save in redis
FOLLOWING_BENEFIT = 0.5
LAST_GOOD_CREATOR_BENEFIT = 0.5
CREATOR_BENEFIT = 1
RANDOM_MEAN, RANDOM_VARIANCE = 0, 0.5
MAX_CREATORS_ON_FEED = 4
POSITIVE_RATING_THRESHOLD = 0.75
RATINGS_MAPPING = {0: 0.9, 1: 0.6, 2: 0.3, 3: 0.15}
NUMBER_INIT_PREFERENCES = 30
PREFERENCES = ['entertainment', 'comedy', 'daily life', 'storytelling', 'arts', 'music', 'fashion beauty',
            'health fitness sport', 'sports', 'diy', 'true crime', 'fiction', 'dating', 'parenting', 'food', 'travel', 
            'languages', 'nature', 'history', 'religion', 'society', 'culture', 'education', 'science', 'career', 'business', 
            'tech', 'finance investing', 'politics', 'news']
GOOD_CREATORS = ["61e7b9286600ea002ea8514d", "62365ca19a1d9536f8afb350", "61e5c362e63580002e097613", "61eb7d26870ee4000450292c", 
                 "61fed4a140c09e0004d3c141", "6204813aa0cccde69355b9af", "6204813aa0cccde69355b9af", "6237b14631ff2267226355ad", 
                 "620733c9a0cccde693696c18", "623e839297d010e000dac917", "623f4a8e630af2cc432a7b66", "61f8a9eee8e4c100048e1a00",
                 "6240e09903e30fc1e613cb1b", "623e503997d010e0005df2f6"]
GOOD_CREATORS_BENEFIT = 10
OPTION = "AVG"
OWN_FEED = ["622ac77943313100046821f8", "6236aedcf73c99000419aa2f", "62369b4098451b0004efb860", "6237ee1b90f6cd0004dcec98", 
            "6238c16fc5bd0b000451d556", "62389aafc5bd0b000451d47e", "623b1d3c26a984000403e880", "623a82254ed8033d5c31fa8b", 
            "623c668575cd1a00040c6e0c", "623fc53f351b080004a2c2c9", "623f6817351b080004a2c246", "623f64a1351b080004a2c1f0"]
AFTER_ONBOARDING_FEED = ["625f3bcd832fbe00043b7788", "625df9273589fe00041aab20", "6254d298d1d8220004fe8314", "6254952ad1d8220004fe81ff", "624a5fa3a3088c0004787c26"]


def get_user_x(user_id):
    """
    gets embedding for user with id [user_id]
    """
    return list(db.users.find({"_id": ObjectId(user_id)}))[0]["initEmbedding"]


def get_content_pool(user_id, redis_ids, redis_last_date):
    """
    Returns a content pool of ids to consider and their repective embeddings
    """
    # get ids of seen ledges; omitting the ones that the user might still listen to
    listened_ids = [l["audio"] for l in db.ratings.find({"user": ObjectId(user_id), "$or": [{"rating": {"$gte": 0.7}}, {"duration": {"$lte": 100}}]})]

    # get ids of blocked users
    blocked_users = [b["to"] for b in db.blocks.find({"from": ObjectId(user_id)})]

    # get ids of blocked content
    blocked_ids = [a["_id"] for a in db.audios.find({"user": {"$in": blocked_users}}, {"_id": 1})]

    redis_last_date = '2022-01-01 00:00:00.000+05:00' if redis_last_date == None else redis_last_date
    # Need to add in-house audios
    # nonlistened_pool = list(
    #     db.audios.find({"_id": {"$nin": list(set(listened_ids) | set(blocked_ids))}, 
    #                     "isVisible": True, 
    #                     "rss": {"$exists": 0}}, 
    #     {"wordEmbedding": 1})
    # )
    nonlistened_pool = list(
        db.audios.find({"_id": {"$nin": list(set(listened_ids) | set(blocked_ids) | set(redis_ids))}, 
                        "isVisible": True, 
                        "wordEmbedding": {"$exists": 1},
                        "rss": {"$exists": 0},
                        "updatedAt": {"$gte": datetime.fromisoformat(str(redis_last_date))}}, 
        {"wordEmbedding": 1})
    )
    listened_ids = [str(id) for id in listened_ids]
    

    # For people who have something in redis, make sure that we get rid of any deleted audios
    if redis_ids:
        found_pool = [elem["_id"] for elem in db.audios.find({"_id": {"$in": redis_ids}, "isVisible": True}, {"_id": 1})]
        to_be_deleted_redis_ids = [str(redis_id) for redis_id in redis_ids if redis_id not in found_pool]
        if len(to_be_deleted_redis_ids) != 0:
            r.hdel("user:" + user_id + ":scores", *to_be_deleted_redis_ids)
        redis_ids = [str(redis_id) for redis_id in redis_ids if redis_id in found_pool]
    return nonlistened_pool, listened_ids, redis_ids


def get_content_pool_from_specific_audios(user_id, specific_audios):
    """
    Returns a content pool of ids to consider and their repective embeddings
    """
    # get ids of seen ledges
    listened_ids = [l["audio"] for l in db.ratings.find({"user": ObjectId(user_id), "audio": {"$in": specific_audios}})]

    # get ids of blocked users
    blocked_users = [b["to"] for b in db.blocks.find({"from": ObjectId(user_id)})]

    # get ids of blocked content
    blocked_ids = [a["_id"] for a in db.audios.find({"user": {"$in": blocked_users}, "audio": {"$in": specific_audios}}, {"_id": 1})]

    nonlistened_pool = list(
        db.audios.find({"_id": {"$nin": list(set(listened_ids) | set(blocked_ids))}, 
                        "_id": {"$in": specific_audios},
                        "isVisible": True, 
                        "wordEmbedding": {"$exists": 1}}, 
        {"wordEmbedding": 1})
    )
    return nonlistened_pool

def diversity_threshold_check(creators, user_id):
    """
    This function makes sure that too many of the audios from the same creator don't show up within one feed
    """
    creators[user_id] += 1
    if creators[user_id] <= MAX_CREATORS_ON_FEED: 
        return True
    return False


def add_history_benefits(scores, user_id):
    """
    Seeing which audios come from people the user is following and increasing their ranking position
    """

    # Get the creators of the potential audios
    audio_creators = list(db.audios.find({"_id": {"$in": [ObjectId(score) for score in scores.keys()]}}, {"user": 1}))
    audio_creators_set = list(set([creator["user"] for creator in audio_creators]))

    # Select the ones that the user is following
    follows_set = set([follow["to"] for follow in db.follows.find({"from": ObjectId(user_id), "to": {"$in": audio_creators_set}}, {"to": 1})])
    audios_from_follows = [str(audio["_id"]) for audio in audio_creators if audio["user"] in follows_set]
    scores = {k: (v + (FOLLOWING_BENEFIT if k in audios_from_follows else 0)) for k, v in scores.items()}

    # Select the audios of favorite creators form the past (one week period - could be changed)
    one_week_ago = datetime.now() - timedelta(days=7)
    ratings_audios = [rating["audio"] for rating in db.ratings.find({"user": ObjectId(user_id), "rating": {"$gte": POSITIVE_RATING_THRESHOLD}, "createdAt": {"$gte": one_week_ago}}, {"audio": 1})]
    most_common_creators = {k: v for k, v in Counter([audio["user"] for audio in db.audios.find({"_id": {"$in": ratings_audios}}, {"user": 1})]).most_common(4)}
    ratings_creators = {k: RATINGS_MAPPING[list(most_common_creators.keys()).index(k)] for k in list(most_common_creators.keys())}
    audios_from_ratings = {str(audio["_id"]): ratings_creators[audio["user"]] for audio in audio_creators if audio["user"] in list(most_common_creators.keys())}
    scores = {k: (v + (audios_from_ratings[k] if k in list(audios_from_ratings.keys()) else 0)) for k, v in scores.items()}

    # Find the audios from the last positive rating's creator
    audio_last_good_creator = [rating["audio"] for rating in db.ratings.find({"user": ObjectId(user_id), "rating": {"$gte": POSITIVE_RATING_THRESHOLD}}, {"audio": 1}).sort("listenedAt", -1).limit(1)]
    # Make sure that we account for people who do not have any previos listening history
    if len(audio_last_good_creator) != 0:
        audio_last_good_creator = audio_last_good_creator[0]
        last_good_creator = list(db.audios.find({"_id": ObjectId(audio_last_good_creator)}, {"user": 1}))[0]["user"]
        audios_last_creator = [str(audio["_id"]) for audio in audio_creators if audio["user"]==last_good_creator]
        scores = {k: (v + (LAST_GOOD_CREATOR_BENEFIT if k in audios_last_creator else 0)) for k, v in scores.items()}
    
    # Promote chosen creators (in-house Auledge creators)
    audio_good_creators = [str(audio["_id"]) for audio in db.audios.find({"user": {"$in": [ObjectId(creator) for creator in GOOD_CREATORS]}}, {"_id": 1})]
    scores = {k: (v + (GOOD_CREATORS_BENEFIT if k in audio_good_creators else 0)) for k, v in scores.items()}

    two_days_ago = datetime.now() - timedelta(days=2)
    hot_feed = db.ratings.aggregate([
        {
            '$match': { 
                'createdAt': {'$gte': two_days_ago},
                'rating': {'$gte': 0.05}
            }
        }, 
        { 
            '$group': { 
                '_id': "$audio", 
                'rating': {'$sum': '$rating'}
            }
        }
    ])
    hot_feed = {str(audio["_id"]): audio["rating"] for audio in hot_feed}
    rated_audios = [audio for audio in hot_feed.keys()]
    scores = {k: (v + (hot_feed[k] if k in rated_audios else 0)) for k, v in scores.items()}

    sorted_scores = {k: v + np.random.normal(RANDOM_MEAN, RANDOM_VARIANCE) for k, v in sorted(scores.items(), key=lambda item: -item[1])[:1000]}
 
    # Restricting the number of audios from the same creators on the feed
    creators = {str(creator): 0 for creator in audio_creators_set}
    audio_creators = {str(item["_id"]): str(item["user"]) for item in audio_creators}
    sorted_scores = {item: sorted_scores[item] for item in sorted_scores.keys() if diversity_threshold_check(creators, audio_creators[item])}

    # Selecting the top 100 audios
    sorted_scores = {k: v for k, v in sorted(sorted_scores.items(), key=lambda item: -item[1])[:100]}
    sorted_scores_keys = [ObjectId(id) for id in sorted_scores.keys()]
    return sorted_scores_keys


def is_new_user(user_id):
    """
    This function checks if the users has joined within the past 24 hours
    """
    one_day_ago = datetime.now() - timedelta(hours=24)
    try:
        print(list(db.users.find({"_id": ObjectId(user_id), "createdAt": {"$gte": one_day_ago}}, {"_id": 1}))[0]["_id"])
        return True
    except:
        return False


def get_sorted_content(mongo_scores, unseen_redis_scores, user_id, seen_redis_scores):
    """
    Return sorted content ids
    Delete listened/anniying ids from redis and anything that goes above 1000 audio ids
    """
    scores = dict(mongo_scores, **unseen_redis_scores)
    # try:
    #     if len(seen_redis_scores) != 0:
    #         r.hdel("user:" + user_id + ":scores", *seen_redis_scores.keys())
    #     if len(scores) >= 1000:
    #         scores_to_be_deleted = dict(sorted(scores.items(), key = itemgetter(1), reverse = True)[1000:])
    #         # r.hdel("user:" + user_id + ":scores", *scores_to_be_deleted.keys())
    # except Exception as e:
    #     print(e)
    
    # This block of code actually benefits the audios from the account a  user is following
    sorted_scores_keys = add_history_benefits(scores, user_id)

    listened_pool = list(
        db.audios.find({"_id": {"$nin": sorted_scores_keys}, 
                        "isVisible": True, 
                        "wordEmbedding": {"$exists": 1},
                        "rss": {"$exists": 0}}, 
        {"_id": 1})
    )
    listened_pool = [elem["_id"] for elem in listened_pool]
    sorted_scores_keys += listened_pool

    return sorted_scores_keys


def get_redis_scores(user_id):
    """
    Gets all the cached scores (corresponding to audios) for a certain user and the timestamp when those scores were generate
    """
    # try:
    #     redis_last_date = r.get("user:" + user_id + ":lastUpdateDate")
    #     redis_last_date = '2022-01-01 00:00:00.000+05:00' if redis_last_date == None else redis_last_date
    # except Exception as e: 
    #     redis_last_date = '2022-01-01 00:00:00.000+05:00'
    #     print(e)

    # # This part of code removes the creators that were just blocked from redis scores
    # blocked_users = [elem["to"] for elem in 
    #     db.blocks.find({"createdAt": {"$gte": datetime.fromisoformat(str(redis_last_date))}}, {"to": 1})]
    # if len(blocked_users) != 0: 
    #     blocked_audios = [str(elem["_id"]) for elem in db.audios.find({"user": {"$in": blocked_users}}, {"_id": 1})]
    #     if len(blocked_audios) != 0:
    #         try:
    #             r.hdel("user:" + user_id + ":scores", *blocked_audios)
    #         except Exception as e:
    #             print(e)

    # try:
    #     redis_scores = r.hgetall("user:" + user_id + ":scores")
    # except Exception as e: 
    #     redis_scores = {}
    #     print(e)
    # redis_ids = redis_scores.keys()
    redis_scores, redis_ids, redis_last_date = [], [], '2022-01-01 00:00:00.000+05:00'
    return redis_scores, redis_ids, redis_last_date


def get_user_preference_vector(user_id):
    """
    Gets user's init embedding and converts it to the the average of word embeddings
    """
    user_x = get_user_x(user_id)
    if (user_x == [0]*NUMBER_INIT_PREFERENCES):
        user_x[26] = 1
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
        r.set("user:" + user_id + ":lastUpdateDate", (datetime.now() - timedelta(hours=1)).isoformat())
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
    It also doesn't showthe audios that were shown in the last feed (in the first 15 audios)
    """
    seen_feeds = [feed["discover"][:TOP_FEED_THRESHOLD] for feed in db.feeds.find({"user": ObjectId(user_id)})][:-1]
    try:
        last_feed = [feed["discover"][:LAST_FEED_THRESHOLD] for feed in db.feeds.find({"user": ObjectId(user_id)}).sort("createdAt", -1).limit(1)][0]
    except IndexError:
        print("This user has no previos feeds")
        last_feed = []
    annoying_audio_ids = [str(audio_id) for feed in seen_feeds for audio_id in feed]
    cnt = Counter(annoying_audio_ids)
    annoying_audio_ids = [audio_id for audio_id, occurences in cnt.items() if occurences >= ANNOYANCE_THRESHOLD]
    return annoying_audio_ids, last_feed


def calculate_scores_for_audios(user_preferences, nonlistened_audios_embeddings):
    if OPTION == "AVG":
        scores = {str(episode_id): np.dot(user_preferences, episode_embedding) for episode_id, episode_embedding in nonlistened_audios_embeddings.items()}
    elif OPTION == "MAX":
        scores = {str(episode_id): np.max([np.dot(user_preference, episode_embedding) for user_preference in user_preferences]) for episode_id, episode_embedding in nonlistened_audios_embeddings.items()}
    # Get 1000 best audios
    scores = dict(sorted(scores.items(), key = itemgetter(1), reverse = True)[:REDIS_THRESHOLD])
    return scores

import time
def get_feed(user_id):
    """
    gets the discovery feed for user with id [user_id]
    """
    start = time.time()
    # Get user's pereferences
    user_preferences = get_user_preference_vector(user_id)

    # Get the data cached in Redis; ids = ids of cached audios; scores = ids & scores
    redis_scores, redis_ids, redis_last_date = get_redis_scores(user_id)
    
    # Query from DB everything besides the ids cached in Redis
    nonlistened_pool, listened_ids, redis_ids = get_content_pool(user_id, [ObjectId(id) for id in redis_ids], redis_last_date)

    # Make sure that we don't take into account the redis scores of audios that were for some reason deleted
    if len(redis_scores) != len(redis_ids):
        redis_scores = {key: value for key, value in redis_scores.items() if ObjectId(key) in redis_ids}

    # Filter out the items that were previously seen too many times on the top of the feed - annoying
    # annoying_audio_ids, last_feed = filter_annoying_audios(user_id)
    annoying_audio_ids, last_feed = [], []

    # Filter only unseen scores from redis; and unseen word Embedidngs from mongo; calculate scores for mongo audios
    nonlistened_redis_scores = {str(key): float(redis_scores[str(key)]) for key in redis_ids if (key not in listened_ids) and (key not in (annoying_audio_ids + last_feed))}
    listened_redis_scores = {key: float(redis_scores[str(key)]) for key in redis_ids if (key in listened_ids) or (key in annoying_audio_ids)}
    nonlistened_audios_embeddings = {audio["_id"]: audio["wordEmbedding"] for audio in nonlistened_pool if audio["_id"] not in annoying_audio_ids}
    mongo_scores = calculate_scores_for_audios(user_preferences, nonlistened_audios_embeddings)

    is_new_user_bool = is_new_user(user_id)
    if is_new_user_bool:
        first_time_audios = [audio for audio in db.audios.find({"_id": {"$in": [ObjectId(id) for id in AFTER_ONBOARDING_FEED]}}, {"_id": 1})]
        first_audios_pool = get_content_pool_from_specific_audios(user_id, [ObjectId(audio["_id"]) for audio in first_time_audios])
        first_time_nonlistened_audios_embeddings = {audio["_id"]: audio["wordEmbedding"] for audio in first_audios_pool if audio["_id"] not in annoying_audio_ids + last_feed}
        first_time_audio_scores = calculate_scores_for_audios(user_preferences, first_time_nonlistened_audios_embeddings)
        sorted_first_time_audios_scores_keys = [ObjectId(id) for id in first_time_audio_scores.keys()]
        first_time_feed = get_data(sorted_first_time_audios_scores_keys)
        order_dict = {_id: index for index, _id in enumerate(sorted_first_time_audios_scores_keys)}
        first_time_feed.sort(key=lambda x: order_dict[x["_id"]])
        first_time_feed = [elem for elem in first_time_feed if str(elem["_id"]) in OWN_FEED] + [elem for elem in first_time_feed if str(elem["_id"]) not in OWN_FEED]

    # Send the new scores to redis
    send_to_redis(user_id, mongo_scores)

    # Rank audios and take the first 100
    sorted_scores_keys = get_sorted_content(mongo_scores, nonlistened_redis_scores, user_id, listened_redis_scores)

    # Query the data for those 100 audios from Mongo
    feed = get_data(sorted_scores_keys)
    # Sort data in the same order
    order_dict = {_id: index for index, _id in enumerate(sorted_scores_keys)}
    feed.sort(key=lambda x: order_dict[x["_id"]])
    
    if is_new_user_bool:
        first_time_ids = [elem["_id"] for elem in first_time_feed]
        feed = first_time_feed + [elem for elem in feed if elem["_id"] not in first_time_ids]
    stop = time.time()
    print(stop-start)

    # print(feed)
    
    # Removed Julia's party audio
    # feed = [elem for elem in feed if elem["_id"] != ObjectId('62389aafc5bd0b000451d47e')]

    return feed


def update_feed(user_id, feed_name, feed):
    """
    update feed named [feed_name] for user with id [user_id]
    """

    values = {}
    values[feed_name] = [item["_id"] for item in feed]
    values["user"] = ObjectId(user_id)
    values["createdAt"] = datetime.now(timezone.utc)
    db.feeds.insert_one(values) 


# TODO: change and expand the number of first time audios
# TODO: Add random noise to first-time predictions
# TODO: HDEL command bug
