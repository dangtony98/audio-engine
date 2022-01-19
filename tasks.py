from celery import Celery
import os
from pydub import AudioSegment
from urllib.request import Request, urlopen
from bson.objectid import ObjectId
import os
import speech_recognition as sr
from client import db
import pickle
import numpy as np

app = Celery()
app.conf.update(broker_url=os.environ['REDIS_URL'],
                result_backend=os.environ['REDIS_URL'], redis_max_connections=20, broker_transport_options = {
    'max_connections': 20,
}, broker_pool_limit=None)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
with open(DIR_PATH + '/src/services/word_embeddings/embeddings_twitter.pickle', 'rb') as handle:
    embeddings_dict = pickle.load(handle)

@app.task
def background_transcribe(audio_ids):
    for audio_id in audio_ids:
        print("Transcribing an audio...")
        sounds = get_audio([audio_id])

        # Update/Insert the transcription into DB
        transcribe_audio(sounds, [audio_id])


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
                    if text == "":
                        text = "UNK"
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                os.remove(audio_ids[sound_num] + str(i) + ".wav")
        embedding = calculate_embedding(text)
        print("sending the data...")
        db.audios.update_one(
            {"_id": ObjectId(audio_ids[sound_num])}, 
            {"$set": {"transcription": text, "wordEmbedding": embedding}},
            upsert=True
        )
        print("Done!")


def calculate_embedding(text):
    print("Calculating embeddings...")
    embedding = np.mean([embeddings_dict[word] for word in text.split() if word in embeddings_dict], axis=0).tolist()
    print("Done calculating embeddings.")
    if embedding != embedding:
        embedding = [0] * 25
    return embedding


# TODO: Delete stopwords