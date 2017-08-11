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

    return stemmed

def has_Number(s):
    return any(i.isdigit() for i in s)


# importing text files from bbc dataset
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
word_dict = {}

for category in categories:
    path = "/Users/Sanghun_Kim/Dropbox/Work/SmartTitle/Code/batch1/bbc-fulltext/bbc/" + category
    files = listdir(path)
    for file in files:
        file_path = join(path, file)
        with open(file_path, 'r', encoding="ISO-8859-1") as f:
            content = f.read().strip()
            processed_content = process_words(content)
            if category in word_dict.keys():
                for token in processed_content:
                    word_dict[category].append(token)
            else:
                word_dict[category] = processed_content

