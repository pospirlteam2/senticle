import numpy as np
import pandas as pd
import tensorflow as tf
import random
import pickle
from soynlp.tokenizer import NounLMatchTokenizer


####################################################
# cut words function                               #
####################################################
def cut(contents):
    results = []
    for content in contents:
        words = content.split()

        # print(words)
        result = []
        for word in words:
            result.append(word)

        results.append(' '.join([token for token in result]))

    return results


    # results = []
    # for content in contents:
    #     words = content.split()
    #     result = []
    #     for word in words:
    #         result.append(word)
    #     results.append([token for token in result])
    #
    # return results


####################################################
# token words function                             #
####################################################
def tokenize(contents):
    return contents.split(' ')

def get_token_id(token, vocab):
    if token in vocab:
        return vocab[token]
    else:
        return 0

def build_input(data, vocab):

    def get_onehot(index, size):
        onehot = [0] * size
        onehot[index] = 1
        return onehot

    print('building input')
    result = []
    temp = []
    for i in range(len(data)):
        temp = data[i].split()

        sequence = [get_token_id(t, vocab) for t in temp]

        # while len(sequence) > 0:
        #     seq_seg = sequence[:800]
        #     sequence = sequence[800:]
        #
        #     padding = [1] *(800 - len(seq_seg))
        #     seq_seg = seq_seg + padding
        #
        #     result.append((seq_seg, get_onehot(d, 2)))
    #
    # return result

        if len(sequence) > 0:
            if len(sequence) > 1400:
                result.append((sequence[:1400], get_onehot(i, 2)))
            else:
                padding = [1] * (1400 - len(sequence))
                sequence = sequence + padding

                result.append((sequence, get_onehot(i, 2)))

    return result

####################################################
# divide train/test set function                   #
####################################################
def divide(x, y, train_prop):
    random.seed(1234)
    x = np.array(x)
    y = np.array(y)
    tmp = np.random.permutation(np.arange(len(x)))
    x_tr = x[tmp][:round(train_prop * len(x))]
    y_tr = y[tmp][:round(train_prop * len(x))]
    x_te = x[tmp][-(len(x)-round(train_prop * len(x))):]
    y_te = y[tmp][-(len(x)-round(train_prop * len(x))):]
    return x_tr, x_te, y_tr, y_te


####################################################
# making input function                            #
####################################################
def make_vocab(documents, max_document_length):
    # tensorflow.contrib.learn.preprocessing 내에 VocabularyProcessor라는 클래스를 이용
    # 모든 문서에 등장하는 단어들에 인덱스를 할당
    # 길이가 다른 문서를 max_document_length로 맞춰주는 역할
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)  # 객체 선언
    x = np.array(list(vocab_processor.fit_transform(documents)))
    ## 텐서플로우 vocabulary processor
    # Extract word:id mapping from the object.
    # word to ix 와 유사
    vocab_dict = vocab_processor.vocabulary_._mapping
    vocabulary = vocab_processor.vocabulary_
    # Sort the vocabulary dictionary on the basis of values(id).
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    # Treat the id's as index into list and create a list of words in the ascending order of id's
    # word with id i goes at index i of the list.
    # print(list(zip(*sorted_vocab)))


    vocabulary = list(list(zip(*sorted_vocab))[0])
    return x, vocabulary, len(vocab_processor.vocabulary_)

    # vocab = dict()
    # vocab['UNK'] = 0
    # vocab['PAD'] = 1
    # for t in documents:
    #     for tokens in t:
    #         if tokens not in vocab:
    #             vocab[tokens] = len(vocab)
    #     # print(t)
    # return  vocab, len(vocab)

####################################################
# savee vocabulary                                 #
####################################################
def save_vocab(filename, documents, max_document_length):
    # tensorflow.contrib.learn.preprocessing 내에 VocabularyProcessor라는 클래스를 이용
    # 모든 문서에 등장하는 단어들에 인덱스를 할당
    # 길이가 다른 문서를 max_document_length로 맞춰주는 역할
    # vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)  # 객체 선언
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)  # 객체 선언
    x = np.array(list(vocab_processor.fit_transform(documents)))
    ### 텐서플로우 vocabulary processor
    # Extract word:id mapping from the object.
    # word to ix 와 유사
    vocab_dict = vocab_processor.vocabulary_._mapping
    # Sort the vocabulary dictionary on the basis of values(id).
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    # Treat the id's as index into list and create a list of words in the ascending order of id's
    # word with id i goes at index i of the list.

    with open(filename, 'w', encoding='utf-8') as f:
        for i in vocab_dict:
            f.write('%s\t%d\n' %(i, vocab_dict[i]))

    # vocabulary = list(list(zip(*sorted_vocab))[0])
    # return x, vocabulary, len(vocab_processor.vocabulary_)

####################################################
# make output function                             #
####################################################
def make_output(points, threshold):
    results = np.zeros((len(points),2))
    for idx, point in enumerate(points):
        if point > threshold:
            results[idx,0] = 1
        else:
            results[idx,1] = 1
    return results

####################################################
# check maxlength function                         #
####################################################
def check_maxlength(contents):
    max_document_length = 0
    for document in contents:
        document_length = len(document.split())
        if document_length > max_document_length:
            max_document_length = document_length
    return max_document_length

####################################################
# loading function                                 #
####################################################
def loading_rdata(data_path):
    # R에서 title과 contents만 csv로 저장한걸 불러와서 제목과 컨텐츠로 분리
    # write.csv(corpus, data_path, fileEncoding='utf-8', row.names=F)
    corpus = pd.read_csv(data_path)
    corpus = corpus.dropna()
    contents = corpus.text
    points = corpus.num
    contents = contents.values.tolist()
    points = points.values.tolist()

    return contents, points


def load_vocab(filename):
    result = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ls = line.split('\t')
            result[ls[0]] = int(ls[1])

    return result

def model_tokenize(contents):
    with open('./nouns_sk.data', 'rb') as f:
    # 포스코 모델
    # with open('./nouns_sk.data', 'rb') as f:

        nouns = pickle.load(f)

    l_match_tokenizer = NounLMatchTokenizer(nouns)

    return l_match_tokenizer.tokenize(contents)


def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False