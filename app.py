from logging import error
from flask import Flask, jsonify, request
from numpy import NaN, e
from pymongo import MongoClient
import json
import os
from client import db, client

app = Flask(__name__)
# if os.environ.get("ENV") == "production":
#     client = MongoClient(os.environ.get("MONGO_PRODUCTION_URI"), tls=True, tlsAllowInvalidCertificates=True)
#     db = client.audio
# else:
#     client = MongoClient(os.environ.get("MONGO_DEVELOPMENT_URI"), tls=True, tlsAllowInvalidCertificates=True)
#     db = client.audio_testing

from src.services.train_xs import *
from src.services.get_discover import *
from src.services.transcribe import *
from src.services.get_prof_recs import *
from src.services.retrain_prof_recommender import *
from src.services.update_creator_embedding import *
from src.services.search import *
from src.middleware.middleware import *
from tasks import background_transcribe

app.wsgi_app = Middleware(app.wsgi_app)


@app.route("/train-xs", methods=["POST"])
def train_xs():
    """
    train all audio and user embeddings where ratings exist by non-zero users
    goal: train accurate latent audio embeddings

    TO-DO 1: disregard ratings where user embeddings are the zero vector to obtain
    precise audio embeddings

    TO-DO 2: add eval
    """

    # train
    ratings, userIds, audioIds = get_ratings_userids_audioids()
    users_P, audios_Q = solve_Q_P(audioIds, userIds, ratings, 2)
    ps_xs = extract_embeddings(userIds, users_P)
    qs_xs = extract_embeddings(audioIds, audios_Q)

    # bulk write
    bulkWriteEmbeddings(ps_xs, db.users)
    bulkWriteEmbeddings(qs_xs, db.audios)
    return json.dumps({"success": True}), 200, {"ContentType": "application/json"}

embeddings_dict = load_embeddings()
@app.route("/get_discover/<string:user_id>", methods=["GET"])
def get_discover(user_id):
    """
    return the "discover" feed for user with id [user_id]
    update the "discover" feed for user with id [user_id] in DB
    """
    # get feed
    feed = get_feed(user_id, embeddings_dict)

    # update feed in DB
    update_feed(user_id, "discover", feed)

    # return cleaned feed
    return jsonify(clean_output(feed))


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    insert the audio text transcription into DB
    """
    request_data = request.get_json()
    audio_ids = request_data['audio_ids']
    for audio_id in audio_ids:
        background_transcribe.delay([audio_id])

    return json.dumps({"success": True}), 200, {"ContentType": "application/json"}


@app.route("/update_creator_embeddings/<string:user_id>", methods=["POST"])
def update_creator_embedding(user_id):
    """
    Update creator embedding every time the user uploads a new audio
    """
    creator_embedding = generate_creator_embedding(user_id)
    
    push_embedding_to_db(user_id, creator_embedding)
    return json.dumps({"success": True}), 200, {"ContentType": "application/json"}


@app.route("/retrain_prof_recommender", methods=["POST"])
def retrain_prof_recommender():
    """
    Retrain the model for creators' profiles recommendations.
    """
    train_model()

    return json.dumps({"success": True}), 200, {"ContentType": "application/json"}


@app.route("/get_prof_recs/<string:user_id>", methods=["GET"])
def get_prof_recs(user_id):
    """
    Get profile recommendations (during the onboarding step)
    """
    creator_ids, final_dataset_X = get_dataset(user_id)

    creator_recs = get_creator_recs(creator_ids, final_dataset_X)
 
    return jsonify(creator_recs), 200, {"ContentType": "application/json"}


@app.route("/search/<string:user_id>/<string:query>", methods=["GET"])
def search(user_id, query):
    """
    Get search suggestions based on a user's query
    """
    search_results = get_search_audio_resuls(query)

    # Push to the DB
    push_search_to_db(user_id, query, search_results)
    return jsonify(search_results), 200, {"ContentType": "application/json"}


@app.route("/pickle", methods=["POST"])
def pickle():
    pickle_word_embeddings()

# import tensorflow as tf
# class MaskedEmbeddingsAggregatorLayer(tf.keras.layers.Layer):
#     def __init__(self, agg_mode='sum', **kwargs):
#         super(MaskedEmbeddingsAggregatorLayer, self).__init__(**kwargs)
#         if agg_mode not in ['sum', 'mean']:
#             raise NotImplementedError('mode {} not implemented!'.format(agg_mode))
#         self.agg_mode = agg_mode
    
#     @tf.function
#     def call(self, inputs, mask=None):
#         masked_embeddings = tf.ragged.boolean_mask(inputs, mask)
#         # tf.print(mask)
#         # tf.print(masked_embeddings)
#         if self.agg_mode == 'sum':
#             aggregated =  tf.reduce_sum(masked_embeddings, axis=1)
#         elif self.agg_mode == 'mean':
#             aggregated = tf.reduce_mean(masked_embeddings, axis=1)
#         return aggregated
    
#     def get_config(self):
#         # this is used when loading a saved model that uses a custom layer
#         return {'agg_mode': self.agg_mode}

# class L2NormLayer(tf.keras.layers.Layer):
#         def __init__(self, **kwargs):
#             super(L2NormLayer, self).__init__(**kwargs)
        
#         @tf.function
#         def call(self, inputs, mask=None):
#             if mask is not None:
#                 inputs = tf.ragged.boolean_mask(inputs, mask).to_tensor()
#             return tf.math.l2_normalize(inputs, axis=-1)

#         def compute_mask(self, inputs, mask):
#             return mask


# @app.route("/train_discover_dnn", methods=["POST"])
# def train_discover_dnn():
#     # Getting all the users 
#     import pandas as pd
#     users = pd.DataFrame(db.users.find({}, {"_id": 1, "initEmbedding": 1, "birthday": 1}))
#     users["age"] = users['birthday'].apply(lambda x: compute_age(x))
#     users.rename(columns={'_id': 'user'}, inplace=True)
#     print(users)

#     # ratings["audio"] = ratings["audio"].factorize()[0] + 1
#     # TODO: 1, 2, or 3?
#     # NUM_CLASSES = ratings["audio"].max() + 3

#     # Getting the ids of all the audios and mapping them to numbers 
#     audios = [element["_id"] for element in db.audios.find({}, {"_id": 1})]
#     print(audios)
#     audio2audio_encoded = {x: i for i, x in enumerate(audios)}
#     audioencoded2audio = {i: x for i, x in enumerate(audios)}
#     NUM_CLASSES = len(audios)
#     # print(audio2audio_encoded)

#     # Getting the ratings and sorting them in the order of listening timestamps
#     ratings = pd.DataFrame(db.ratings.find({}, {"audio": 1, "user": 1, "rating": 1}).sort("listenedAt"))
#     print(ratings)

#     # Ecncoding ObkectIds into numbers 
#     ratings["audio"] = ratings["audio"].map(audio2audio_encoded)
#     audios = list(ratings["audio"])
#     print(audios)

#     # Groupping everything by users and aggregating the ratings 
#     ratings = ratings.drop(columns="_id").groupby('user').agg(list).reset_index()
#     ratings = users.drop(columns='birthday').merge(ratings, on="user", how='right')
#     # ratings['rating'] = ratings['rating'].apply(lambda x: x if type(x) is list else []) # NaN 처리

#     # Creating all historical sequences of audios and ratings
#     ratings["audio"] = ratings['audio'].apply(lambda x: [x[:i+1] for i in range(len(x))])
#     ratings["rating"] = ratings['rating'].apply(lambda x: [x[:i+1] for i in range(len(x))])

#     # Spliting all the sequence into separate data points 
#     final_ratings = pd.DataFrame({"user": [], "initEmbedding": [], "age": [], "audio": [], "rating": []})
#     for index, row in ratings.iterrows(): 
#         for elem in range(len(row["audio"])):
#             final_ratings.loc[len(final_ratings.index)] = [row["user"], row["initEmbedding"], row["age"], row["audio"][elem], row["rating"][elem]]
#     ratings = final_ratings
#     print(ratings)
    
#     # Creating the column that we need to predict 
#     ratings["predict"] = ratings["audio"].apply(lambda x: x[-1])
#     ratings['audio'] = ratings['audio'].apply(lambda x: x[:-1])
#     ratings['rating'] = ratings['rating'].apply(lambda x: x[:-1])

#     # Splitting into training vs testing data
#     # train_data = ratings[(ratings.index <= len(ratings.index) - 2)]
#     # test_data = ratings[(ratings.index >= len(ratings.index) - 1)]
#     test_data = ratings[(ratings.index <= 0)]
#     train_data = ratings[(ratings.index >= 1)]

#     # Imnitializing constant for model training 
#     EMBEDDING_DIMS = 16
#     DENSE_UNITS = 64
#     DROPOUT_PCT = 0.0
#     ALPHA = 0.0
#     LEARNING_RATE = 0.003

#     import tensorflow as tf
#     import datetime
#     import os
#     input_watch_hist = tf.keras.Input(shape=(None, ), name='watch_hist')
#     input_watch_hist_time = tf.keras.layers.Input(shape=(None,), name='watch_hist_time')
#     input_age = tf.keras.layers.Input(shape=(1,), name='age')
#     input_init_embeddings = tf.keras.layers.Input(shape=(12,), name='input_init_embeddings')


#     #--- layers
#     features_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS, 
#                                                 mask_zero=True, trainable=True, name='features_embeddings')
#     labels_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS, 
#                                                 mask_zero=True, trainable=True, name='labels_embeddings')

#     avg_embeddings = MaskedEmbeddingsAggregatorLayer(agg_mode='mean', name='aggregate_embeddings')

#     dense_1 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_1')
#     dense_2 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_2')
#     dense_3 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_3')
#     l2_norm_1 = L2NormLayer(name='l2_norm_1')

#     dense_output = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax, name='dense_output')

#     #--- features
#     features_embeddings = features_embedding_layer(input_watch_hist)
#     l2_norm_features = l2_norm_1(features_embeddings)
#     avg_features = avg_embeddings(l2_norm_features)

#     labels_watch_embeddings = labels_embedding_layer(input_watch_hist_time)
#     l2_norm_watched = l2_norm_1(labels_watch_embeddings)
#     avg_watched = avg_embeddings(l2_norm_watched)

#     labels_watch_embeddings = labels_embedding_layer(input_init_embeddings)
#     l2_norm_watched = l2_norm_1(labels_watch_embeddings)

#     concat_inputs = tf.keras.layers.Concatenate(axis=1)([avg_features, avg_watched, input_age, input_init_embeddings])

#     # Dense Layers
#     dense_1_features = dense_1(concat_inputs)
#     dense_1_relu = tf.keras.layers.ReLU(name='dense_1_relu')(dense_1_features)
#     dense_1_batch_norm = tf.keras.layers.BatchNormalization(name='dense_1_batch_norm')(dense_1_relu)

#     dense_2_features = dense_2(dense_1_relu)
#     dense_2_relu = tf.keras.layers.ReLU(name='dense_2_relu')(dense_2_features)
#     # dense_2_batch_norm = tf.keras.layers.BatchNormalization(name='dense_2_batch_norm')(dense_2_relu)

#     dense_3_features = dense_3(dense_2_relu)
#     dense_3_relu = tf.keras.layers.ReLU(name='dense_3_relu')(dense_3_features)
#     dense_3_batch_norm = tf.keras.layers.BatchNormalization(name='dense_3_batch_norm')(dense_3_relu)
#     outputs = dense_output(dense_3_batch_norm)

#     #Optimizer
#     optimiser = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

#     #--- prep model
#     model = tf.keras.models.Model(
#         inputs=[input_watch_hist, 
#                 input_watch_hist_time, 
#                 input_age,
#                 input_init_embeddings],
#         outputs=[outputs]
#     )
#     logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#     model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['acc'])

#     history = model.fit([tf.keras.preprocessing.sequence.pad_sequences(train_data['audio'], maxlen=12) + 1e-10,
#            tf.keras.preprocessing.sequence.pad_sequences(train_data['rating'], maxlen=12, dtype=float) + 1e-10, 
#            train_data['age'],
#            np.array(list(train_data['initEmbedding']), dtype=int)], 
#            train_data['predict'].values, steps_per_epoch=1, epochs=50)

#     pred = model.predict([tf.keras.preprocessing.sequence.pad_sequences(test_data['audio'], maxlen=12) + 1e-10,
#            tf.keras.preprocessing.sequence.pad_sequences(test_data['rating'], maxlen=12, dtype=float) + 1e-10, 
#            test_data['age'],
#            np.array(list(test_data['initEmbedding']), dtype=int)
#            ])

#     model.save("candidate_generation.h5")

#     # Generating candidates for a given user
#     print(pred[0])
#     import heapq
#     import operator
#     candidates = list(zip(*heapq.nlargest(50, enumerate(pred[0]), key=operator.itemgetter(1))))[0]
#     print(candidates)
#     # N = 50
#     # k = np.sort((-pred).argsort()[:,:N])
#     # print(k)
#     # k = [np.array(list(map(audioencoded2audio, k[0])))]
#     # k = [audioencoded2audio[x] for x in k[0]]
#     # print(k)



#     return json.dumps({"success": True}), 200, {"ContentType": "application/json"}


# model = tf.keras.models.load_model(
#         'candidate_generation.h5',
#         custom_objects={
#             'L2NormLayer':L2NormLayer,
#             'MaskedEmbeddingsAggregatorLayer':MaskedEmbeddingsAggregatorLayer
#         }
#     )


# @app.route("/get_discover_dnn/<string:user_id>", methods=["GET"])
# def get_discover_dnn(user_id):
#     # load candidate_generation 
#     import timeit
#     # def load():
    
#     # loop = 10
#     # result = timeit.timeit(lambda: load(), number=loop)
#     # print(result / loop)
#     import pandas as pd
#     user_info = pd.DataFrame(db.users.find({"_id": ObjectId(user_id)}, {"initEmbedding": 1, "birthday": 1}))
#     ratings_info = list(db.ratings.find({"user": ObjectId(user_id)}, {"rating": 1, "audio": 1}))
#     print(ratings_info)
#     # print(ratings_info)
#     audios_r = [elem["audio"] for elem in ratings_info]
#     ratings = [elem["rating"] for elem in ratings_info]

#     audios = [element["_id"] for element in db.audios.find({}, {"_id": 1})]
#     # print(audios)
#     audio2audio_encoded = {x: i + 1 for i, x in enumerate(audios)}
#     audioencoded2audio = {i + 1: x for i, x in enumerate(audios)}

#     audios_encoded = [audio2audio_encoded[elem] for elem in audios_r]
#     # print(audios_encoded)

#     user_info["age"] = user_info['birthday'].apply(lambda x: compute_age(x))

#     pred = model.predict([tf.keras.preprocessing.sequence.pad_sequences([audios_encoded], maxlen=12) + 1e-10,
#            tf.keras.preprocessing.sequence.pad_sequences([ratings], maxlen=12, dtype=float) + 1e-10, 
#            user_info["age"],
#            np.array(list(user_info["initEmbedding"]), dtype=int)
#            ])


#     import heapq
#     import operator
#     a = list(zip(*heapq.nlargest(50, enumerate(pred[0]), key=operator.itemgetter(1))))[0]
#     print(a)

    

    # N = 50
    # k = np.sort((-pred).argsort()[:,:N])
    # print(k)
    # k = [audioencoded2audio[x + 1] for x in k[0]]
    # print(k)

    # result = list(db.audios.find({"_id": {"$in": k}}))

    # # print(result)

    # return json.dumps({"success": True}), 200, {"ContentType": "application/json"}


if __name__ == '__main__': 
    app.run(debug=True)