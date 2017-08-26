# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:16:50 2017

@author: Sanghun_Kim
"""

# Natural Language ToolKit (NLTK)

import nltk
nltk.download()

from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

twitter_samples.fileids()
neg_tweets = twitter_samples.strings('negative_tweets.json')
print(neg_tweets[0])

example_text = 'To center the data (make it have zero mean and unit standard error), you subtract the mean and then divide the result by the standard deviation. You do that on the training set of data. But then you have to apply the same transformation to your testing set (e.g. in cross-validation), or to newly obtained examples before forecast.'
sent_tokens = sent_tokenize(example_text)
print(sent_tokens)
print(len(sent_tokens))

word_tokens = word_tokenize(example_text)
print(word_tokens)
print(len(word_tokens))


# Using Stopwords
stop_words = stopwords.words('english')
filtered_sent = [w for w in word_tokens if w not in stop_words]
print(filtered_sent)


# Using PorterStemmer for stemming
ps = PorterStemmer()
stemmed = [ps.stem(w) for w in word_tokens]
print(stemmed)


# Using POS Tagging
'''
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
'''


# 