from pydub import AudioSegment
from urllib.request import Request, urlopen
from bson.objectid import ObjectId
from app import db
import os
import speech_recognition as sr
# from googletrans import Translator
# import googletrans
# from flask import Flask, render_template, url_for, request
# from textblob import TextBlob

MAX_API_LENGTH_MS = 30000


def get_audio(audio_ids):
    # url = db.audios.find({"_id": ObjectId(audio_id)})[0]["url"]
    audio_ids = [ObjectId(audio_id) for audio_id in audio_ids]
    urls = [audio["url"] for audio in db.audios.find({"_id": {"$in": audio_ids}})]
    print(urls)
    sounds = []
    for url in urls: 
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        mp3file = urlopen(req)
        with open('./initial.mp3','wb') as output:
            output.write(mp3file.read())

        sounds.append(AudioSegment.from_mp3("./initial.mp3"))
        print("check", len(sounds))
        os.remove("initial.mp3")
    return sounds


def transcribe_audio(sounds, audio_ids):
    r = sr.Recognizer()

    print(len(sounds))
    print(audio_ids)
    for sound_num in range(len(sounds)): 
        text = ""
        # Need to split the audio into pieces of 30 sec max because of the API limits
        for i in range(len(sounds[sound_num])//MAX_API_LENGTH_MS + 1):
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

    # TRYING LANGUAGE DETECTION, NEED TO CLEAN UP 

    # # detect_language(text, audio_id)
    # translator = Translator()
    # print('A')
    # # print(translator.detect(text))
    # print(translator.detect('hello').lang)
    # print('A')
    # blobline = TextBlob('hello') 
    # detected_language = blobline.detect_language() 

    # db.audios.update_one(
    #     {"_id": ObjectId(audio_id)}, 
    #     {"$set": {"language": str(detected_language)}},
    #     upsert=True
    # )
    # import requests
    # import urllib.parse
    # url = "https://google-translate1.p.rapidapi.com/language/translate/v2/detect"

    # payload = "q=English%20is%20hard%2C%20but%20detectably%20so"
    # headers = {
    #     'content-type': "application/x-www-form-urlencoded",
    #     'accept-encoding': "application/gzip",
    #     'x-rapidapi-host': "google-translate1.p.rapidapi.com",
    #     'x-rapidapi-key': "8cc0863199msh3c9519925606cf2p12df3cjsn0da250e85d03"
    #     }

    # response = requests.request("POST", url, data="q=" + urllib.parse.quote('hello'), headers=headers)

    # print(response.json()["data"]["detections"][0][0]["language"])


# def detect_language(text, audio_id):
#     translator = Translator()
#     print('A')
#     # print(translator.detect(text))
#     language = translator.detect('hello').lang
#     print('A')

#     db.audios.update_one(
#         {"_id": ObjectId(audio_id)}, 
#         {"$set": {"language": str(language)}},
#         upsert=True
#     )