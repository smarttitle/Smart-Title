# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "Amazon plans to weave together its online business and physical stores by turning its Prime membership program into a Whole Foods rewards program, providing additional savings to customers. Amazon Prime is a $99-a-year service that gives customers faster free shipping, video streaming and other benefits. Whole Foods’ private-label products will be available through Amazon’s online services and Amazon lockers that will be installed in some Whole Foods markets. Customers will also be able to return online orders to Amazon through the lockers. “We’re determined to make healthy and organic food affordable for everyone,” Jeff Wilke, the executive who runs Amazon’s consumer businesses, said on Thursday in an announcement about the changes. “Everybody should be able to eat Whole Foods Market quality.”"

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)