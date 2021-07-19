from ..variables import PREF_TO_INDEX
from bson.objectid import ObjectId
import numpy as np
import cvxpy as cp
from cvxpy import *
from app import db
from pymongo import UpdateOne


def get_rs_ps_qs():
    """
    return [rs], [ps], [qs]:
    - [rs] is a list of ratings
    - [ps] is a list of user ids
    - [qs] is a list of audio ids
    """
    targets = list(db.ratings.find({}))

    rs = np.array(list(map(lambda x: x["rating"], targets)))
    ps = list(map(lambda x: str(x["user"]), targets))
    qs = list(map(lambda x: str(x["audio"]), targets))
    return rs, ps, qs


def preprocess(preferences):
    theta = [0] * len(PREF_TO_INDEX)

    for preference in preferences:
        index = PREF_TO_INDEX[preference]
        theta[index] = 1

    return theta


def get_P(ps):
    """
    create matrix P by getting [ps] preferences and creating embeddings
    """
    p_dict = {}  # stores map [user id]: [initial embedding]

    ps_uniq = list(map(lambda x: ObjectId(x), set(ps)))
    users = list(db.users.find({"_id": {"$in": ps_uniq}}))

    index = 0
    for user in users:
        if str(user["_id"]) not in p_dict:
            # case: user is not in p_dict
            p = user["initEmbedding"]
            p_dict[str(user["_id"])] = {"p": p, "index": index}
            index += 1

    P = np.array([p_dict[pid]["p"] for pid in ps])
    return P


def create_P_Q(psqs):
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


def solve_Q(P, qs, rs):
    """
    solve for qs embeddings (audio)
    """

    Q = create_P_Q(qs)

    constraints = []

    obj = cp.Minimize(sum_squares(diag(P @ Q.T) - rs) + 0.001 * sum_squares(Q))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return P, Q.value, rs


def solve_P(Q, ps, rs):
    """
    solves for ps embeddings (user)
    """
    P = create_P_Q(ps)

    constraints = []
    obj = cp.Minimize(sum_squares(diag(P @ Q.T) - rs) + 0.001 * sum_squares(P))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return P.value, Q, rs


def solve_Q_P(qs, ps, rs, loops):
    """
    solve for [qs] and [ps] embeddings given [rs]
    """
    P = get_P(ps)
    Q = None
    for _ in range(0, loops):
        P, Q, rs = solve_Q(P, qs, rs)
        P, Q, rs = solve_P(Q, ps, rs)

    return P, Q


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
