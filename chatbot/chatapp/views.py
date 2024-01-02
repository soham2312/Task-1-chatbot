from django.shortcuts import render
import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from django.http import HttpResponse, JsonResponse
from django.template import loader

# Initialize WordNetLemmatizer for lemmatization
lemmatizer = WordNetLemmatizer()

# Load the Q/A file
try:
    QandA_file = json.loads(open('./../Question_and_Answers.json').read())
except Exception as e:
    # Handle JSON file loading error
    print(f"Error loading Q/A file: {e}")
    QandA_file = {}

# Load the pickle files containing preprocessed data
try:
    words = pickle.load(open('./../words.pkl', 'rb'))
    classes = pickle.load(open('./../classes.pkl', 'rb'))
except Exception as e:
    # Handle pickle file loading error
    print(f"Error loading pickle files: {e}")
    words, classes = [], []

# Load the pre-trained chatbot model
try:
    model = load_model('./../chatbot_model.h5')
except Exception as e:
    # Handle model loading error
    print(f"Error loading chatbot model: {e}")
    model = None

def index(request):
    # Render the chat.html template
    template = loader.get_template('chat.html')
    return HttpResponse(template.render({}, request))

def get(request, method='GET'):
    # Handle the GET request for chat interactions
    if request.method == 'GET':
        user_text = request.GET.get('msg', '')
        try:
            bot_response = get_chat_response(user_text)
            return JsonResponse({'response': bot_response})
        except Exception as e:
            # Handle chat response generation error
            print(f"Error generating chat response: {e}")
            return JsonResponse({'error': 'Error generating chat response'})
    else:
        return JsonResponse({'error': 'Invalid request method'})

def get_chat_response(userText):
    # Tokenize and lemmatize user input
    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    # Create a bag of words representation for the input sentence
    def bag_of_words(sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    # Predict the intent class based on the input sentence
    def predict_class(sentence):
        p = bag_of_words(sentence)
        if model is not None:
            res = model.predict(np.array([p]))[0]
            ERROR_THRESHOLD = 0.25
            results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
            return return_list
        else:
            # Handle case where model is not loaded
            return []

    # Get a response based on the predicted intent
    def get_response(intents_list, intents_json):
        if not intents_list or not intents_json:
            # Handle empty or invalid intents list or JSON
            return "Sorry, I'm having trouble understanding right now."
        
        tag = intents_list[0]['intent']
        list_of_intents = intents_json.get('questions_and_answers', [])
        for i in list_of_intents:
            if i.get('tag') == tag:
                responses = i.get('responses', [])
                if responses:
                    return random.choice(responses)
                else:
                    # Handle case where responses list is empty
                    return "I don't have a response for that at the moment."
        # Handle case where tag is not found in intents
        return "I'm not sure how to respond to that."

    # Predict intent and get the chatbot response
    ints = predict_class(userText)
    res = get_response(ints, QandA_file)
    return res
