from ..variables import PREF_TO_INDEX
from bson.objectid import ObjectId
from bson import json_util
import numpy as np
from cvxpy import *
from app import db


def get_user_x(user_id):
    """
    gets embedding for user with id [user_id]
    """
    return list(db.users.find({"_id": ObjectId(user_id)}))[0]["embedding"]


def get_content_pool(user_id):
    """
    Returns a content pool of ids to consider and their repective embeddings
    """

    seen_ids = [l["audio"] for l in db.listens.find({"user": ObjectId(user_id)})]

    unseen_pool = list(db.audios.find({"_id": {"$nin": seen_ids}}))
    seen_pool = list(db.audios.find({"_id": {"$in": seen_ids}}))

    unseen_pool_xs = [a["embedding"] for a in unseen_pool]
    seen_pool_xs = [a["embedding"] for a in seen_pool]

    return seen_pool, seen_pool_xs, unseen_pool, unseen_pool_xs


def get_sorted_content(user_x, content_pool, content_pool_xs):
    """
    Return sorted content ratings
    """

    if not content_pool:
        # case: [content_pool] is empty
        return content_pool

    ratings = np.dot(np.array(content_pool_xs), np.array(user_x))
    sort_idx = ratings.argsort()

    return list(np.array(content_pool)[sort_idx[::-1]])


def clean_output(pool):
    """
    clean [pool] to be compatible with Flask's jsonify
    """

    def clean(item):
        item["_id"] = str(item["_id"])
        item["user"] = str(item["user"])
        return item

    return [clean(item) for item in pool]


def get_feed(user_id):

    user_x = get_user_x(user_id)
    seen_pool, seen_pool_xs, unseen_pool, unseen_pool_xs = get_content_pool(user_id)

    sorted_seen_pool = get_sorted_content(user_x, seen_pool, seen_pool_xs)
    sorted_unseen_pool = get_sorted_content(user_x, unseen_pool, unseen_pool_xs)

    feed = sorted_unseen_pool + sorted_seen_pool
    return feed


def update_feed(user_id, feed_name, feed):
    """
    update feed named [feed_name] for user with id [user_id]
    """

    filter = {"user": ObjectId(user_id)}

    values = {}
    values[feed_name] = [item["_id"] for item in feed]
    values["user"] = ObjectId(user_id)
    new_values = {"$set": values}

    db.feeds.update_one(filter, new_values, upsert=True)
