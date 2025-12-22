import random
import json
import pickle
import numpy as np
import tensorflow as tf
import requests

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('D:\chatbot\create_chatbot_using_python-main\create_chatbot_using_python-main\intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# --- 1. PREPARE DATA ---
corpus = []  # Sentences
tags = []    # Labels

for doc in documents:
    pattern = [lemmatizer.lemmatize(w.lower()) for w in doc[0]] #this is your 'pattern' list, ex: ['how', 'are', 'you']
    corpus.append(' '.join(pattern)) # Convert list of words back to sentence
    tags.append(doc[1])

# --- 2. VECTORIZE ---
vectorizer = CountVectorizer(vocabulary=words)
trainX = vectorizer.transform(corpus).toarray()

lb = LabelBinarizer()
lb.fit(classes)
trainY = lb.transform(tags)

trainX, trainY = shuffle(trainX, trainY, random_state=42)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')

# --- PART 2: DYNAMIC WEATHER FUNCTION ---

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


