from logging import error
from flask import Flask, jsonify
from numpy import NaN, e
from pymongo import MongoClient
import json
import os

app = Flask(__name__)
client = MongoClient(
    os.environ.get("MONGO_URI"), tls=True, tlsAllowInvalidCertificates=True
)

db = client.audio

from src.services.train_xs import *
from src.services.get_discover import *
from src.services.transcribe import *
from src.services.get_prof_recs import *
from src.services.retrain_prof_recommender import *
from src.services.update_creator_embedding import *
from src.services.search import *
from src.middleware.middleware import *

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


@app.route("/get_discover/<string:user_id>", methods=["GET"])
def get_discover(user_id):
    """
    return the "discover" feed for user with id [user_id]
    update the "discover" feed for user with id [user_id] in DB
    """

    # get feed
    feed = get_feed(user_id)

    # update feed in DB
    update_feed(user_id, "discover", feed)

    # return cleaned feed
    return jsonify(clean_output(feed))


@app.route("/transcribe/<string:audio_id>", methods=["POST"])
def transcribe(audio_id):
    """
    insert the audio text transcription into DB
    """
    
    sound = get_audio(audio_id)

    # Update/Insert the transcription into DB
    transcribe_audio(sound, audio_id)

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


@app.route("/search/<string:query>", methods=["GET"])
def search(query):
    """
    Get search suggestions based on a user's query
    """
    search_results = get_search_audio_resuls(query)
    return jsonify(search_results), 200, {"ContentType": "application/json"}


if __name__ == '__main__': 
    app.run(debug=True)