# from ..variables import PREF_TO_INDEX
# from bson.objectid import ObjectId
# import numpy as np
# import cvxpy as cp
# from cvxpy import *
# from index import db


# def get_x(theta, y):
#     """
#     return a trained audio embedding [x] given [theta] and [y]
#     - param [theta]: a 2d-array of encoded user preferences
#     - param [y]: a 1d-array of ratings
#     """

#     x = cp.Variable(12)  # the embedding to solve for

#     constraints = []
#     obj = cp.Minimize((1 / 2) * sum_squares(theta @ x - y) + (1 / 2) * sum_squares(x))
#     prob = cp.Problem(obj, constraints)
#     prob.solve()
#     return x.value


# def get_theta_y(audio_id):
#     """
#     return (theta, y) for audio with id [audio_id]
#     - param [audio_id]: the audio to get theta, y for
#     - [theta] is a 2d-array of encoded user preferences using [pp_pref_batch]
#     - [y] is a 1d-array of ratings
#     """
#     target_ratings = list(db.ratings.find({"audio": ObjectId(audio_id)}))
#     target_users = list(
#         db.users.find({"_id": {"$in": list(map(lambda r: r["user"], target_ratings))}})
#     )

#     preferences_batch = list(map(lambda u: u["preferences"], target_users))

#     theta = pp_pref_batch(preferences_batch)
#     y = list(map(lambda r: r["rating"], target_ratings))
#     return (theta, y)


# def pp_pref_instance(preferences):
#     """
#     preprocess [preferences]
#     - param [preferences]: an array of topic preferences
#     - precondition: [preferences] is a unique array of valid topics (i.e. keys in
#       PREF_TO_INDEX)
#     """

#     theta = [0] * len(PREF_TO_INDEX)

#     for preference in preferences:
#         index = PREF_TO_INDEX[preference]
#         theta[index] = 1

#     return theta


# def pp_pref_batch(preferences_batch):
#     """
#     preprocess [preferences_batch]
#     - param [preferences_batch]: an array of preferences where each preference
#       is a key in [PREF_TO_INDEX]
#     - precondition: [preferences_batch] contains unique arrays of valid topics

#     """
#     return np.array(list(map(pp_pref_instance, preferences_batch)))
