from django.shortcuts import render
import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from django.http import HttpResponse
from django.template import loader
lemmatizer = WordNetLemmatizer()
# Load the Q/A file
QandA_file = json.loads(open('./../Question_and_Answers.json').read())

# Load the pickle files
words = pickle.load(open('./../words.pkl','rb'))
classes = pickle.load(open('./../classes.pkl','rb'))

    # Load the model
model = load_model('./../chatbot_model.h5')

def index(request):
    template = loader.get_template('chat.html')
    return HttpResponse(template.render({}, request))

def get(request):
    userText = request.GET.get('msg')
    return get_chat_response(userText)

def get_chat_response(userText):
    def clean_up_sentence(sentence):
        # Tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # Lemmatize each word - create base word, in attempt to represent related words
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    # Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bag_of_words(sentence):
        # Tokenize the pattern
        sentence_words = clean_up_sentence(sentence)
        # Bag of words
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    # Assign 1 if current word is in the vocabulary position
                    bag[i] = 1
        return np.array(bag)

    def predict_class(sentence):
        # Filter below  threshold predictions
        p = bag_of_words(sentence)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        # Sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        
        return return_list

    def get_respons(intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['questions_and_answers']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    
    ints = predict_class(userText)
    res = get_respons(ints, QandA_file)
    return res
