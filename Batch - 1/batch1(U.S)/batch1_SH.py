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
import numpy as np
import re


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
word_dict = {}

for category in categories:
    path = "/Users/Sanghun_Kim/Dropbox/Work/SmartTitle/Code/batch1/bbc-fulltext/bbc/" + category
    files = listdir(path)
    token_dict = {}
    for file in files:
        file_path = join(path, file)
        with open(file_path, 'r', encoding="ISO-8859-1") as f:
            content = f.read().strip()
            processed = process_words(content)
            
            token_dict[file] = processed
    
    word_dict[category] = token_dict

# Calculating TF-IDF scores 
tfidf_dict = {}
keyword_dict = {}

tfvectorizer = TfidfVectorizer(tokenizer= tokenizer, encoding='utf-8', stop_words='english', lowercase=True)

for category in categories:
    tfidf_dict[category] = tfvectorizer.fit_transform(word_dict[category].values())
    
    # Sorting scores and mapping top 10 keywords to keyword_dict by corresponding categories
    for i in range(0, len(word_dict[category])):
        scores = tfidf_dict[category][i]
        feature_array = np.array(tfvectorizer.get_feature_names())   
        score_sorted = np.argsort(scores.toarray().flatten())[::-1]
        
        n = 10
        top_n = list(feature_array[score_sorted][:n])
        
        if category in keyword_dict.keys():
            keyword_dict[category].append(top_n)
        else:
            keyword_dict[category] = [top_n]