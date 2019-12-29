import tflearn
import tensorflow as tf
import random
from src.lib.important import bag_of_words, extract_from_json, train_to_bow

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

import json
# Load Data
with open('data/intents.json') as file:
    data = json.load(file)

# Extract data from json
words, labels, train_x, train_y = extract_from_json(data)

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# Transform data into bag of words
training, output = train_to_bow(words, labels, train_x, train_y)

# Training Model
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model/model.tflearn")

# Chat app
def chat():
    print('ketik "quit" untuk berhenti')
    while True:
        inp = input(">>> ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
          print('Maaf kami tidak dapat memproses pertanyaan anda. Silahkan ganti pertanyaan lain.')

#chat()

