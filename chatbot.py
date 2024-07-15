import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
import warnings
import tensorflow as tf
from gtts import gTTS
import os
import pygame
import logging
from flask import Flask, request, jsonify, render_template
from textblob import TextBlob  
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

app = Flask(__name__,template_folder='templates',static_folder='static')
lemmatizer = WordNetLemmatizer()
logging.basicConfig(level=logging.INFO)

class modelTrain:
    def __init__(self):
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!']

    def loadIntents(self, intents_path=''):
        data_file = open(intents_path).read()
        intents = json.loads(data_file)
        return intents

    def preprocess_save_Data(self, intents):
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                self.documents.append((w, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        print(len(self.documents), " documents ")
        print(len(self.classes), " classes ", self.classes)
        print(len(self.words), " unique lemmatized words ", self.words)

        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

        return self.words, self.classes

    def prepareTrainingData(self, words, classes):
        training = []
        output_empty = [0] * len(classes)

        for doc in self.documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)

        train_x = np.array(list(training[:, 0]))
        train_y = np.array(list(training[:, 1]))

        return train_x, train_y

    def createModel(self, train_x, train_y, epochs=200, batch_size=8, save_path='model.model'):
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        hist = model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1)
        model.save(save_path, hist)
        print("Model Successfully Created and saved")
        return model

class modelPredict:
    def __init__(self, intents_path='data.json', model_path='model.model'):
        self.intents_path = intents_path
        self.model = model_path

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words, show_details=False):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("Found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence, model, error_threshold=0.25):
        ERROR_THRESHOLD = error_threshold
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
        p = self.bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints, intents_json):
        list_of_intents = intents_json['intents']
        result = None  # Initialize result to a default value

        if ints:
            tag = ints[0]['intent']
            for intent in list_of_intents:
                if intent['tag'] == tag:
                    result = random.choice(intent['responses'])
                    break

        if not result:  # Fallback if no intent was matched or if the result is still None
            for intent in list_of_intents:
                if intent['tag'] == 'fallback':
                    result = random.choice(intent['responses'])
                    break

        return result
    
    def chatbot_response(self, msg):
        model = load_model(self.model)
        intents = json.loads(open(self.intents_path).read())
        ints = self.predict_class(msg, model)
        res =  self.getResponse(ints, intents)
        return res
    
    def correct_spelling(self, sentence):
        corrected_sentence = TextBlob(sentence).correct()
        return str(corrected_sentence)

# Initialize trainer
trainer = modelTrain()
intents = trainer.loadIntents('data.json')
words, classes = trainer.preprocess_save_Data(intents)
train_x, train_y = trainer.prepareTrainingData(words, classes)

# Ensure all elements in train_x have the same shape
max_len = max(len(x) for x in train_x)
from keras.preprocessing.sequence import pad_sequences
train_x = pad_sequences(train_x, maxlen=max_len, padding='post')


model = trainer.createModel(train_x, train_y, save_path='cbv_model.h5')

predictor = modelPredict('data.json', 'cbv_model.h5')

@app.route('/')
def index():
    return render_template('index4.html')
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    # Predefined greetings and farewells
    greetings = {"hi", "hii", "hiii", "hiiii", "hiiiii", "hiiiiii", "hiiiiiii", "hello"}
    farewells = {"bye", "by", "byy", "byyy", "byyyy", "byyyyy", "byee", "byeee", "byeeee", "byeeeee"}

    if user_input.lower() in greetings:
        return jsonify(response="Hello!", images=[])
    elif user_input.lower() in farewells:
        return jsonify(response="Goodbye!", images=[])
    else:
        corrected_input = predictor.correct_spelling(user_input)
        print(f"Original input: {user_input}, Corrected input: {corrected_input}")  # Debug print
        response = predictor.chatbot_response(corrected_input)
        images = []
        if "Cargo pants in 2024 feature updated silhouettes" in response:
            images.append('static/CARGO.jpg')
        elif "another response condition" in response:
            images.append('static/ANOTHER_IMAGE.jpg')
        
        return jsonify(response=response, images=images)


if __name__ == "__main__":
    app.run(port=8080)
