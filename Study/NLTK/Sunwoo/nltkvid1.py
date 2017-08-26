from nltk.tokenize import sent_tokenize, word_tokenize


# tokenizing - word tokenizers.. senetence tokenizers
# lexicon and corporas
# corpora - body of text. ex : medical journals, presidential speeches, English language
# lexicon - words and their means

example_text = "Hello Mr. Smith, how are you doing today? The weather is greate and Python is awesome. The sky is pinkish-blue. You should not eat cardboard."

#print(sent_tokenize(example_text))
#print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)