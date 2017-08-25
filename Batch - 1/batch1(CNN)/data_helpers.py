import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords


def has_Number(s):
    return any(i.isdigit() for i in s)


def tokenizer(text):
    tokens = word_tokenize(text)
    return tokens

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " have", string) # \'ve
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " are", string) # \'re
    string = re.sub(r"\'d", " would", string) # \'d
    string = re.sub(r"\'ll", " will", string) # \'ll
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    # pre-process word dictionary (cleaning non-important words, stemming, stopwords, etc)
    # def process_words(text):

    string = [word for word in re.split("\W+", string)]


    lowered_list = [a.lower() for a in string]

    string = lowered_list;

    stop_words = stopwords.words('english')
    stopped_text = []
    for word in string:
        # get rid of any string containing digits together in the process of filtering stopwords
        if word not in stop_words and (has_Number(word) == False):
            stopped_text.append(word)

    ps = PorterStemmer()
    stemmed = [ps.stem(w) for w in stopped_text]
    string = " ".join(stemmed)
    return string.strip().lower()



def load_data_and_labels(csv_file): # 뉴스 데이터 가져 오기
    dataset = pd.read_csv(csv_file, header=0)

    x_test = dataset['body'].values.tolist()
    y_tmp = dataset['categories'].values.tolist()
    x_test = [clean_str(sent) for sent in x_test]
    y = list()

    for category in y_tmp:
        if category == 'politics':
            y.append([1,0,0,0,0])
        elif category == 'business':
            y.append([0,1,0,0,0])
        elif category == 'entertainment':
            y.append([0,0,1,0,0])
        elif category == 'tech':
            y.append([0,0,0,1,0])
        else:
            y.append([0,0,0,0,1])

    return [x_test,np.array(y)]

################
def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors

#####



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
