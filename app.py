from logging import error
from flask import Flask, jsonify
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
    rs, ps, qs = get_rs_ps_qs()
    P, Q = solve_Q_P(qs, ps, rs, 2)
    ps_xs = extract_embeddings(ps, P)
    qs_xs = extract_embeddings(qs, Q)

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


# @app.route("/transcribe/<string:audio_id>", methods=["POST"])
# def transcribe(audio_id):
#     """
#     insert the audio text transcription into DB
#     """
    
#     sound = get_audio(audio_id)

#     # Update/Insert the transcription into DB
#     transcribe_audio(sound, audio_id)

#     return json.dumps({"success": True}), 200, {"ContentType": "application/json"}