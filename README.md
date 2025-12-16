# 🤖 Python AI Chatbot with Dynamic Skills

A smart chatbot built using **Python**, **TensorFlow**, and **NLTK**. 
It uses Deep Learning (Neural Networks) to understand user intent and can perform dynamic actions like fetching real-time weather data using APIs.

## 🚀 Features
* **Intent Recognition:** Uses a Neural Network to classify user queries into tags (e.g., greeting, question, help).
* **Dynamic Responses:** Can fetch **Real-Time Weather** for your current location using IP geolocation.
* **Easy Customization:** All conversation patterns are stored in a simple `intents.json` file.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** TensorFlow / Keras
* **NLP:** NLTK (Natural Language Toolkit)
* **API Integration:** Requests module (for Weather & IP Geolocation)

## 📂 Project Structure
* `chatbot.py` - The main script to run the bot.
* `new.py` - The training script. Run this if you edit `intents.json`.
* `intents.json` - Contains the training data (patterns and responses).
* `words.pkl` & `classes.pkl` - Saved data from training.
* `chatbot_model.h5` - The trained Neural Network model.
