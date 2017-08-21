# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 20:34:36 2017

@author: Kihoon
@title: Batch1 - TfidfVectorizer
"""

from os import listdir
from os.path import join
from collections import Counter
from nltk.corpus import stopwords
import nltk
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
#    filtered_tokens = [w for w in tokens if not w in stopwords.words('english')]
    stems = stem_tokens(tokens, stemmer)
    return stems

#categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
categories = ['business']
word_dict = {}
token_dict = {}

for category in categories:
    path = './bbc/' + category
    files = listdir(path)
#    word_dict[category] = {}
    for file in files:
        file_path = join(path, file)
        with open(file_path, 'r', encoding="ISO-8859-1") as f:
            content = f.read().strip()
            content = re.sub('[^A-Za-z]+', ' ', content)
        txt = content.lower()
        txt = txt.translate(string.punctuation)
#        token_dict[file] = tokenize(content)
        token_dict[file] = txt

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())