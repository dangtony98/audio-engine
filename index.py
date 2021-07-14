from flask import Flask, json, jsonify
from pymongo import MongoClient
from dotenv import dotenv_values
import json

config = dotenv_values(".env")
app = Flask(__name__)
client = MongoClient(config["MONGO_URI"], tls=True, tlsAllowInvalidCertificates=True)

db = client.audio

# import modules
from src.services.create_audio_xs import *
from src.services.get_discover import *
from src.middleware.middleware import *

# @app.route("/create-embedding/<string:audioId>")
# def create_audio_x(audioId):
#     # trains a single embedding for audio with id [audioId]

#     (theta, y) = get_theta_y(audioId)
#     x = get_x(theta, y)

#     return jsonify(x.tolist())

app.wsgi_app = Middleware(app.wsgi_app)


@app.route("/create-audio-xs/")
def create_audio_xs():
    """
    train all audio and user embeddings where ratings exist
    update embeddings in DB
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


@app.route("/get_discover/<string:user_id>")
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
