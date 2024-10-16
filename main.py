
### Author: Faiaz Amin Khan & Humayra Jahan Himika ###
### AI Lab Project ###

#importing necessary library
import random
import json
import pickle
import numpy as np


import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

#initiate lemmatizer
lemmatizer = WordNetLemmatizer()

#Load all the necessary files
intents = json.loads(open('intents.json').read())

words_in_pkl = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

#function to clean sentence from unnecessary characters
def clean_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word)  for word in words]
    return words

#create bag of words from intent
def bag_of_words(sentence):
    words= clean_sentence(sentence)
    bag = [0] * len(words_in_pkl)
    for w in words:
        for i, word in enumerate(words_in_pkl):
            if word == w:
                bag[i] = 1

    return np.array(bag)

# predict the class of the sentence
def class_prediction(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# prepare response and return to print
def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("|============== Welcome to Class Equiry Chatbot System! ==============|")
print("|============================== Feel Free ============================|")
print("|================================== To ===============================|")
print("|=============== Ask your any query about our classes ================|")
print("|===============****************ClassBot**************================|")
while True:
    message_input = input("| You: ")
    if message_input == "bye" or message_input == "Goodbye":
        intentions = class_prediction(message_input)
        response = get_response(intentions, intents)
        print("| Bot:", response)
        print("|===================== The Program End here! See you soon =====================|")
        exit()

    else:
        intentions = class_prediction(message_input)
        response = get_response(intentions, intents)
        print("| Bot:", response)
