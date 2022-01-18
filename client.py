import os
from pymongo import MongoClient


if os.environ.get("ENV") == "production":
    client = MongoClient(os.environ.get("MONGO_PRODUCTION_URI"), tls=True, tlsAllowInvalidCertificates=True)
    db = client.audio
else:
    client = MongoClient(os.environ.get("MONGO_DEVELOPMENT_URI"), tls=True, tlsAllowInvalidCertificates=True)
    db = client.audio_testing