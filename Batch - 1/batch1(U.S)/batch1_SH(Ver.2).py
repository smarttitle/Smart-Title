# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 18:56:32 2017

@author: Sanghun_Kim
@title: Batch 1 - Classifying news categories based on the content of the article
"""

from os import listdir
from os.path import join
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models, similarities
import numpy as np
import re
import pickle


# pre-process word dictionary (cleaning non-important words, stemming, stopwords, etc)
def process_words(text):
    text = [word for word in re.split("\W+", text)]

    stop_words = stopwords.words('english')
    stopped_text = []
    for word in text:
        # get rid of any string containing digits together in the process of filtering stopwords
        if word not in stop_words and (has_Number(word) == False):
            stopped_text.append(word)
    
    ps = PorterStemmer()
    stemmed = [ps.stem(w) for w in stopped_text]
    text = " ".join(stemmed)
    return text

def has_Number(s):
    return any(i.isdigit() for i in s)

def tokenizer(text):
    tokens = word_tokenize(text)
    return tokens
    

# importing text files from bbc dataset
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
txt_dict = {}

for category in categories:
    path = "/Users/Sanghun_Kim/Dropbox/Work/SmartTitle/Code/batch1/bbc-fulltext/bbc/" + category
    files = listdir(path)
    token_dict = {}
    for file in files:
        file_path = join(path, file)
        with open(file_path, 'r', encoding="ISO-8859-1") as f:
            content = f.read().strip()
            processed = process_words(content)
            
            token_dict[category+'-'+file] = processed
						
    txt_dict.update(token_dict)
    
pickle.dump(txt_dict, open('text_dict.p', 'wb'))


# Calculating TF-IDF scores 
tfidf_dict = {}
keyword_dict = {}

tfvectorizer = TfidfVectorizer(tokenizer= tokenizer, encoding='utf-8', stop_words='english', lowercase=True)
tfidf_dict = tfvectorizer.fit_transform(txt_dict.values())

all_keys = list(txt_dict.keys())

for i in range(0, len(txt_dict)):
	scores = tfidf_dict[i]
	feature_array = np.array(tfvectorizer.get_feature_names())
	score_sorted = np.argsort(scores.toarray().flatten())[::-1]
	
	n = 10
	top_n = list(feature_array[score_sorted][:n])
	keyword_dict[all_keys[i]] = top_n
    

# Transform keywords into vector using word2vec_lookup_table
wordvect_lookup = models.Word2Vec.load('word2vec_lookup_table')
len(wordvect_lookup.wv.vocab)  # Vocabulary size of model

# Test for one document
doc_wv = [[wordvect_lookup[keyword] for keyword in keyword_dict['business-001.txt']]]

# Mapping with all docuements in the keyword_dict
for doc in keyword_dict.values():
    doc_wv = [[wordvect_lookup[keyword] for keyword in doc]]










