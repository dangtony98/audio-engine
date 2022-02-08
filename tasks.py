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
with open(DIR_PATH + '/src/services/word_embeddings/stopwords.pickle', 'rb') as handle:
        stopwords = pickle.load(handle)

@app.task
def background_transcribe(audio_ids):
    i = 0
    for audio_id in audio_ids:
        try: 
            i += 1
            print("Transcribing an audio... #" + str(audio_id) + ": " + str(i) + " out of " + str(len(audio_ids)))
            sounds = get_audio(audio_id)

            # Update/Insert the transcription into DB
            transcribe_audio(sounds, audio_id)
        except:
            print("!!!!! Transcription Failed - " + str(audio_id))


MAX_API_LENGTH_MS = 30000


def get_audio(audio_id):
    print(audio_id)
    try:
        urls = [audio["url"] for audio in db.audios.find({"_id": ObjectId(audio_id)}, {"url": 1})]
    except Exception as e:
        # Strange "connection pool paused" error
        print("ERROR", e)
    sounds = []
    for url in urls: 
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        file = urlopen(req)
        try:
            PATH = './initial' + audio_id + '.mp3'
            with open(PATH,'wb') as output:
                output.write(file.read())
                sounds.append(AudioSegment.from_file(PATH))
                os.remove(PATH)
        except:
            PATH = './initial' + audio_id + '.m4a'
            with open(PATH,'wb') as output:
                output.write(file.read())
                sounds.append(AudioSegment.from_file(PATH))
                os.remove(PATH)
    return sounds


def transcribe_audio(sounds, audio_id):
    r = sr.Recognizer()

    for sound_num in range(len(sounds)): 
        text = ""
        # Need to split the audio into pieces of 30 sec max because of the API limits
        NUMBER_OF_SEGMENTS = len(sounds[sound_num])//MAX_API_LENGTH_MS + 1
        if NUMBER_OF_SEGMENTS <= 30:
            for i in range(NUMBER_OF_SEGMENTS):
                print("Segment " + str(i))
                sounds[sound_num][i*MAX_API_LENGTH_MS:(i+1)*MAX_API_LENGTH_MS].export(audio_id + str(i) + ".wav", format="wav")
                    
                with sr.AudioFile(audio_id + str(i) + ".wav") as source:
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
                    os.remove(audio_id + str(i) + ".wav")
            if text != "UNK":
                embedding = calculate_embedding(text)
            print("Sending the data...")
            db.audios.update_one(
                {"_id": ObjectId(audio_id)}, 
                {"$set": {"transcription": text, "wordEmbedding": embedding}},
                upsert=True
            )
            print("Done!")
        else: 
            db.audios.update_one(
                {"_id": ObjectId(audio_id)}, 
                {"$set": {"isVisible": False, "duration": NUMBER_OF_SEGMENTS * 30}},
                upsert=True
            )
            print("Audio is too long! - " + str(audio_id))
    print("All Finished!")


def calculate_embedding(text):
    print("Calculating embeddings...")
    text = " ".join([word for word in text.split() if not word in stopwords])
    embedding = np.mean([embeddings_dict[word] for word in text.split() if word in embeddings_dict], axis=0).tolist()
    print("Done calculating embeddings.")
    if embedding != embedding:
        embedding = [0] * 25
    return embedding
