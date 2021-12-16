from pydub import AudioSegment
from urllib.request import Request, urlopen
from bson.objectid import ObjectId
from app import db
import os
import speech_recognition as sr

MAX_API_LENGTH_MS = 30000


def get_audio(audio_id):
    url = db.audios.find({"_id": ObjectId(audio_id)})[0]["url"]

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    mp3file = urlopen(req)
    with open('./initial.mp3','wb') as output:
        output.write(mp3file.read())

    sound = AudioSegment.from_mp3("./initial.mp3")
    os.remove("initial.mp3")
    return sound


def transcribe_audio(sound, audio_id):
    r = sr.Recognizer()

    text = ""
    # Need to split the audio into pieces of 30 sec max because of the API limits
    for i in range(len(sound)//MAX_API_LENGTH_MS + 1):
        sound[i*MAX_API_LENGTH_MS:(i+1)*MAX_API_LENGTH_MS].export(audio_id + str(i) + ".wav", format="wav")
            
        with sr.AudioFile(audio_id + str(i) + ".wav") as source:
            audio_text = r.listen(source)

            try:
                # using google speech recognition
                text += r.recognize_google(audio_text) + " "
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                text += "UNK"
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
            os.remove(audio_id + str(i) + ".wav")

    db.transcriptions.update_one(
        {"audio_id": ObjectId(audio_id)}, 
        {"$set": {"text": text}},
        upsert=True
    )