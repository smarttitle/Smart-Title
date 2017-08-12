# -*- coding: utf-8 -*-

# use natural language toolkit
import nltk
import newspaper
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import json
import datetime
stemmer = LancasterStemmer()

# 3 classes of training data
import pandas as pd

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
training_data = []

for i in range(333):
    training_data.append({"class":y[i],"sentence": X[i] })

#training_data.append({"class":"Sports", "sentence":"Sports is good"})

print ("%s sentences in training data" % len(training_data))

words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# lemmatizing

lemmatizer = WordNetLemmatizer()
words = [w for w in words if lemmatizer.lemmatize(w)]

# remove stop words
stopwords = nltk.corpus.stopwords.words('english')
words= [w for w in words if w.lower() not in stopwords]

# remove not alphabet words

words = [ w for w in words if w.isalpha()]

# remove duplicates
classes = list(set(classes))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
print ([stemmer.stem(word.lower()) for word in w])
print (training[i])
print (output[i])

import numpy as np
import time

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)

X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
    
# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results


classify(X[33])
for i in range(333):
    classify(X[i]),
    print("answer" + y[i])

classify('''Could this slimy, brightly colored orange goo be a real-life version of the "Flubber" from the 1997 film starring Robin Williams?

Not quite... But while it may look like it's made for children, in reality it is revolutionizing the way soldiers are protected, and could radically reduce the number of head trauma injuries in football.
It's a gel developed by UK-company D3O that acts as both a liquid and a solid. When handled slowly the goo is soft and flexible but the moment it receives an impact, it hardens. It's all thanks to the gel's shock-absorbing properties.
Read: Firefighters can see through smoke with new thermal mask
"If I wrap it around my fingers, it's very soft," Felicity Boyce, a material developer at D3O, told CNN, "but if you hit it with great force, it behaves more like a solid that's absorbing the shock and none of that impact goes through my hand."
&quot;Flubber&quot; featured Robin Williams as a quirky professor who discovers a green, rubbery substance.
"Flubber" featured Robin Williams as a quirky professor who discovers a green, rubbery substance.
With careful blending, the team has been able to use the patented gel in helmets, mobile phone cases, gloves and most notably, armor for football players and those in the military.
"Obviously you couldn't just use this (gel) inside a protection item -- because it's messy, it's sticky, and it's gooey," Boyce said. "So that's the clever part -- incorporating this into something that can be used as an end product and still maintaining those properties of being soft and flexible."
'Endless opportunities'
Developed in 1999, the material rose to prominence during the 2006 Winter Olympics in Torino, when apparel company Spyder used D3O's protective gel for the US and Canadian ski teams' race suits.
Read: Color-changing inks respond to the environment
Since then, it's been used by Usain Bolt in the soles of his shoes at the 2016 Rio Olympics and the company has teamed up with many global brands in sport, electronics, defense and industrial apparel.
Olympic gold medallist Usain Bolt has used D3O&#39;s sport insole, which is designed to improve performance and reduce injury.
Olympic gold medallist Usain Bolt has used D3O's sport insole, which is designed to improve performance and reduce injury.
Floria Antolini, the chief knowledge officer at D3O, said there were endless opportunities for the orange goo.
"When you create something so different than anything on the market, there's a lot of experimentation (that can be done)," she said.
Absorbing impact
American football has become a huge market for the British company, where the gel is incorporated in padding and helmets to absorb the impact of any hits a player receives.
In 2015 alone, there were 271 reported concussions in the NFL, but by using this technology, D3O hopes to dramatically decrease the number of head trauma injuries.
"We're part of the solution and we can definitely contribute to protecting people so that they experience fewer injuries," Antolini said.
Testers drop different weights on the D3O helmets to find out how much force the material can absorb.
Testers drop different weights on the D3O helmets to find out how much force the material can absorb.
To ascertain how much impact can be absorbed by D3O helmets, the company conducts tests on mannequin heads.
"The main testing we're interested in is the impact performance by, dropping different weights on materials, and to measure the force that is transmitted to the material," Boyce said.
"That tells us how well that would protect your body from a particular impact."
D3O claims it can reduce blunt impact by 53 per cent compared to materials like foam.
Protection and comfort
D3O conducts tests on mannequin heads to see how much impact can be absorbed via their helmet padding.
D3O conducts tests on mannequin heads to see how much impact can be absorbed via their helmet padding.
The D3O team has also worked extensively with the US and UK defense forces, police and emergency services to provide protective and comfortable clothes.
Antolini said the company first started looking at how to protect soldiers from impacts and improve everything from their shoes to the padding on their knees, elbows and shoulders -- so that those in combat could hold their positions without becoming uncomfortable.
She added: "While we don't have a material that can stop a bullet, we do have a material that can reduce the amount of trauma that your body would experience if you got shot."
''')

cnn_paper = newspaper.build('http://cnn.com',memoize_articles=False)

news_list = []
ans_list=[]
url_list=[]
for article in cnn_paper.articles:
    if(article.url.find("tech")!=-1):
        article.download()
        article.parse()
        print(article)
        article.text
        url_list.append(article.url)
        news_list.append(article.text)
        ans_list.append("tech")
        
for i in news_list:
    classify(i)
