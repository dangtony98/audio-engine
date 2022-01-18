from celery import Celery
import os
from pydub import AudioSegment
from urllib.request import Request, urlopen
from bson.objectid import ObjectId
import os
import speech_recognition as sr
from pymongo import MongoClient
from client import db, client

app = Celery()
# app.conf.from_object("celery_settings")
app.conf.update(BROKER_URL=os.environ['REDIS_URL'],
                CELERY_RESULT_BACKEND=os.environ['REDIS_URL'])


# if os.environ.get("ENV") == "production":
#     client = MongoClient(os.environ.get("MONGO_PRODUCTION_URI"), tls=True, tlsAllowInvalidCertificates=True)
#     db = client.audio
# else:
#     client = MongoClient(os.environ.get("MONGO_DEVELOPMENT_URI"), tls=True, tlsAllowInvalidCertificates=True)
#     db = client.audio_testing


# from src.services.transcribe import *


@app.task
def background_transcribe(audio_ids):
    print("Start")
    for audio in audio_ids:
        sounds = get_audio([audio])

        # Update/Insert the transcription into DB
        transcribe_audio(sounds, [audio])


MAX_API_LENGTH_MS = 30000


def get_audio(audio_ids):
    audio_ids = [ObjectId(audio_id) for audio_id in audio_ids]
    urls = [audio["url"] for audio in db.audios.find({"_id": {"$in": audio_ids}})]
    
    sounds = []
    for url in urls: 
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        mp3file = urlopen(req)
        with open('./initial.mp3','wb') as output:
            output.write(mp3file.read())

        sounds.append(AudioSegment.from_mp3("./initial.mp3"))
        os.remove("initial.mp3")
    return sounds


def transcribe_audio(sounds, audio_ids):
    r = sr.Recognizer()

    for sound_num in range(len(sounds)): 
        text = ""
        # Need to split the audio into pieces of 30 sec max because of the API limits
        for i in range(len(sounds[sound_num])//MAX_API_LENGTH_MS + 1):
            print("Segment", i)
            sounds[sound_num][i*MAX_API_LENGTH_MS:(i+1)*MAX_API_LENGTH_MS].export(audio_ids[sound_num] + str(i) + ".wav", format="wav")
                
            with sr.AudioFile(audio_ids[sound_num] + str(i) + ".wav") as source:
                audio_text = r.listen(source)

                try:
                    # using google speech recognition
                    text += r.recognize_google(audio_text) + " "
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                    text += "UNK"
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                os.remove(audio_ids[sound_num] + str(i) + ".wav")
        db.audios.update_one(
            {"_id": ObjectId(audio_ids[sound_num])}, 
            {"$set": {"transcription": text}},
            upsert=True
        )
