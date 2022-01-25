from ..variables import PREF_TO_INDEX
from bson.objectid import ObjectId
import numpy as np
import cvxpy as cp
from cvxpy import *
from app import db
from pymongo import UpdateOne


def get_ratings_userids_audioids():
    """
    return [rs], [ps], [qs]:
    - [rs] is a list of ratings
    - [ps] is a list of user ids
    - [qs] is a list of audio ids
    """
    targets = list(db.ratings.find({}))

    ratings = np.array(list(map(lambda x: round(x["rating"], 2), targets)))
    userids = list(map(lambda x: str(x["user"]), targets))
    audioids = list(map(lambda x: str(x["audio"]), targets))

    return ratings, userids, audioids


def preprocess(preferences):
    theta = [0] * len(PREF_TO_INDEX)

    for preference in preferences:
        index = PREF_TO_INDEX[preference]
        theta[index] = 1

    return theta


def get_users_P(userIds):
    """
    create matrix P by getting [ps] preferences and creating embeddings
    """
    users_dict = {}  # stores map [user id]: [initial embedding]

    users_uniq = list(map(lambda x: ObjectId(x), set(userIds)))
    users = list(db.users.find({"_id": {"$in": users_uniq}}))

    index = 0
    for user in users:
        if str(user["_id"]) not in users_dict:
            # case: user is not in p_dict
            users_dict[str(user["_id"])] = {"p": user["initEmbedding"], "index": index}
            index += 1

    P = np.array([users_dict[pid]["p"] for pid in userIds])
    return P


def create_users_P_audios_Q(psqs):
    """
    creates a matrix of variables for optimization
    """
    dict = {}  # stores map [psqs id]: [embedding]

    def populate_dict(item, dict):
        if item in dict:
            # case: user is in p_dict
            return dict[item]
        else:
            # case: user is not in p_dict
            val = cp.Variable(12, nonneg=True)
            dict[item] = val
            return val

    matrix = cp.vstack([populate_dict(item, dict) for item in psqs])
    return matrix


def solve_audios_Q(users_P, audioIds, ratings):
    """
    solve for qs embeddings (audio)
    """

    audios_Q = create_users_P_audios_Q(audioIds)

    constraints = []

    obj = cp.Minimize(sum_squares(diag(users_P @ audios_Q.T) - ratings) + 0.001 * sum_squares(audios_Q))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return users_P, audios_Q.value, ratings


def solve_users_P(audios_Q, userIds, ratings):
    """
    solves for ps embeddings (user)
    """
    users_P = create_users_P_audios_Q(userIds)

    constraints = []
    obj = cp.Minimize(sum_squares(diag(users_P @ audios_Q.T) - ratings) + 0.001 * sum_squares(users_P))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return users_P.value, audios_Q, ratings


def solve_Q_P(audioIds, userIds, ratings, loops):
    """
    solve for [qs] and [ps] embeddings given [rs]
    """
    users_P = get_users_P(userIds)
    audios_Q = None
    for _ in range(0, loops):
        users_P, audios_Q, ratings = solve_audios_Q(users_P, audioIds, ratings)
        users_P, audios_Q, ratings = solve_users_P(audios_Q, userIds, ratings)

    return users_P, audios_Q


def extract_embeddings(psqs, PQ):
    dict = {}  # stores index mappings

    for idx, pq in enumerate(psqs):
        if pq not in dict:
            dict[pq] = idx

    def create_dict(key):
        k = {}
        k["key"] = key
        k["embedding"] = PQ[dict[key]].tolist()
        return k

    x = [create_dict(key) for key in dict]
    print(x)
    return x


def bulkWriteEmbeddings(xs, collection):
    def create_op(e):
        return UpdateOne(
            {"_id": ObjectId(e["key"])},
            {"$set": {"embedding": e["embedding"]}},
            upsert=True,
        )

    operations = list(map(create_op, xs))
    result = collection.bulk_write(operations)
    print(result.bulk_api_result)
    return None
