import random
import json
import pickle
import numpy as np
import nltk
import requests

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r'D:\chatbot\create_chatbot_using_python-main\create_chatbot_using_python-main\intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def get_weather():
    """Fetches weather for the user's current location using IP Geolocation."""
    try:
        # STEP 1: Get location based on IP address
        # We use ip-api.com (Free, no key required) to get lat/lon/city
        loc_response = requests.get("http://ip-api.com/json/")
        loc_data = loc_response.json()

        # Check if the location call was successful
        if loc_data['status'] == 'fail':
            return "Error: Could not determine your location."

        latitude = loc_data['lat']
        longitude = loc_data['lon']
        city = loc_data['city']

        # STEP 2: Get weather for those specific coordinates
        # We inject the variables {latitude} and {longitude} into the URL
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()
        
        temp = weather_data['current_weather']['temperature']
        wind = weather_data['current_weather']['windspeed']
        
        return f"The current temperature in {city} is {temp}°C with a wind speed of {wind} km/h."

    except Exception as e:
        return f"Sorry, I ran into an error: {e}"

# MAPPING: Connect intents to functions
intent_methods = {
    "get_weather": get_weather
}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    sentence_string = " ".join(sentence_words)
    vectorizer = CountVectorizer(vocabulary=words)

    bag = vectorizer.transform([sentence_string]).toarray()[0]
    return bag

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I didn't understand that."
    
    tag = intents_list[0]['intent']
    
    # DYNAMIC CHECK: If the tag matches our specialized functions
    if tag in intent_methods:
        result = intent_methods[tag]()
    else:
        # STATIC FALLBACK: Use JSON responses
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    return result

# --- RUN THE CHAT ---
print("\nWeather Bot is running! (Ask 'How is the weather?')")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)


    
