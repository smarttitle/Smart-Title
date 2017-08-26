# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:35:02 2017

@author: Sanghun_Kim
"""

from gensim import models, similarities
from nltk import word_tokenize

import pickle

# Corpus generation
txt_dict = pickle.load(open('text_dict.p','rb'))
corpus = [word_tokenize(word) for word in txt_dict.values()]


# word2vec model instantiation
wordvect_lookup = models.Word2Vec(corpus, min_count=1, size=24, window=3)
len(wordvect_lookup.wv.vocab)
wordvect_lookup.save('word2vec_lookup_table')
