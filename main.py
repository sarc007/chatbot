# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout

# used a dictionary to represent an intents JSON file
data = {"intents": [
    {"tag": "greeting",
     "patterns": ["Hello", "How are you", "How are you?", "Hi there", "Hi", "Whats up"],
     "responses": ["Howdy Partner!", "Hello", "How are you doing?", "Greetings!", "How do you do?"],
     },
    {"tag": "Store2Door",
     "patterns": ["What is Store2Door Cargo LLC", "What is Store2Door", "What do you do?", "What about you",
                  "What is store2door", "Store2Door", "store2door"],
     "responses": ["We are a freight company that caters to customers such as tourists, expats, students etc. "
                   "to deliver their personal cargo internationally and locally."
                   ],
     },
    {"tag": "Location",
     "patterns": ["Located", "located", "location", "Location"],
     "responses": ["We are based out of Dubai, United Arab Emirates."
                   ],
     },
    {"tag": "Timing",
     "patterns": ["your timings", "your timing", "timings please", "working hours", "working days", "working days and hours"],
     "responses": ["We are open from Sundays through Thursdays from 09.00 am to 06.00 pm "
                   "and on Saturdays 09.00 am to 01.00 pm."
                   ],
     },
    {"tag": "use Store2Door",
     "patterns": ["use Store2Door", "use store2door", "use store to door"],
     "responses": ["Any individual or company who has a personal package to be sent "
                   "from the U.A.E to any part of the world or even within the U.A.E. can use Store2Door."
                   ],
     },

    {"tag": "age",
     "patterns": ["how old are you?", "when is your birthday?", "when was you born?"],
     "responses": ["I am 24 years old", "I was born in 1996", "My birthday is July 3rd and I was born in 1996",
                   "03/07/1996"]
     },
    {"tag": "date",
     "patterns": ["what are you doing this weekend?",
                  "do you want to hang out some time?", "what are your plans for this week"],
     "responses": ["I am available all week", "I don't have any plans", "I am not busy"]
     },
    {"tag": "name",
     "patterns": ["what's your name?", "what are you called?", "who are you?"],
     "responses": ["My name is Kippi", "I'm Kippi", "Kippi"]
     },
    {"tag": "goodbye",
     "patterns": ["bye", "g2g", "see ya", "adios", "cya"],
     "responses": ["It was nice speaking to you", "See you later", "Speak soon!"]
     }
]}
# Each list to create
words = []
classes = []
doc_X = []
doc_Y = []
train_X = None
train_Y = None
lemmatizer = WordNetLemmatizer()
model = None


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def download_data():
    nltk.download("punkt")
    nltk.download("wordnet")


def lemmatize_data():
    global words
    global classes
    global doc_X
    global doc_Y
    # global lemmatizer
    # Loop through all the intents
    # tokenize each pattern and append tokens to words, the patterns and
    # the associated tag to their associated list
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            doc_X.append(pattern)
            doc_Y.append(intent["tag"])

        # add the tag to the classes if it's not there already
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
    # lemmatize all the words in the vocab and convert them to lowercase
    # if the words don't appear in punctuation
    words_loc = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
    # sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
    words = sorted(set(words))
    classes = sorted(set(classes))


def list_training_data():
    # list for training data
    global words
    global classes
    global doc_X
    global doc_Y
    global train_X
    global train_Y
    training = []
    out_empty = [0] * len(classes)

    # creating the bag of words model
    for idx, doc in enumerate(doc_X):
        bow = []
        text = lemmatizer.lemmatize(doc.lower())
        for word in words:
            bow.append(1) if word in text else bow.append(0)
        # mark the index of class that the current pattern is associated
        # to
        output_row = list(out_empty)
        output_row[classes.index(doc_Y[idx])] = 1
        # add the one hot encoded BoW and associated classes to training
        training.append([bow, output_row])
    # shuffle the data and convert it to an array
    random.shuffle(training)
    training = np.array(training, dtype=object)
    # split the features and target labels
    train_X = np.array(list(training[:, 0]))
    train_Y = np.array(list(training[:, 1]))


def train_data():
    global model
    # defining some parameters
    input_shape = (len(train_X[0]),)
    output_shape = len(train_Y[0])
    epochs = 200
    # the deep learning model
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(output_shape, activation="softmax"))
    adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["accuracy"])
    print(model.summary())
    model.fit(x=train_X, y=train_Y, epochs=200, verbose=1)


def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result_pred = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result_pred) if res > thresh]

    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result_resp = random.choice(i["responses"])
            break
    return result_resp


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    download_data()
    lemmatize_data()
    list_training_data()
    train_data()

    while True:
        message = input("")
        intents = pred_class(message, words, classes)
        result_chat = get_response(intents, data)
        print(result_chat)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
