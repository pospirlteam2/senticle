import pandas as pd
from soynlp.tokenizer import NounLMatchTokenizer
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor
import math
# from konlpy.tag import Kkma
import csv
import pickle

# kkma = Kkma()

# import pickle
# from fastText import load_model
# from gensim.models import Word2Vec


# model = load_model('/home/pirl/Downloads/fastText-master/POSCO.bin')


data_path = 'preprocessed_SKhynix.csv'

doc = pd.read_csv(data_path)

contents = []
points = []

for i in range(0, len(doc['text'])):
    if len(str(doc['text'][i])) > 100:
        contents.append(doc['text'][i])
        points.append(doc['num'][i])


# print(kkma.pos(contents[0]))
#
# words = []
#
# for i in contents:
#     temp = i.split(' ')
#     words.append(temp)
#
# word2vec = Word2Vec(words)
#
# print(word2vec.most_similar('철강'))
#
# word_extractor = WordExtractor(
#     min_frequency=100,
#     min_cohesion_forward=0.05,
#     min_right_branching_entropy=0.0
# )
#
# word_extractor.train(contents) # list of str or like
# words = word_extractor.extract()
#
# cohesion_scores = {word:score.cohesion_forward for word, score in words.items()}
# ltokenizer = LTokenizer(scores = cohesion_scores)
#
#
#
# def word_score(score):
#     return (score.cohesion_forward * math.exp(score.right_branching_entropy))
#
# print('단어   (빈도수, cohesion, branching entropy)\n')
# for word, score in sorted(words.items(), key=lambda x:word_score(x[1]), reverse=True)[:30]:
#     print('%s     (%d, %.3f, %.3f)' % (
#             word,
#             score.leftside_frequency,
#             score.cohesion_forward,
#             score.right_branching_entropy
#             )
#          )
#
# for i in words:
#     print(i)
#
# noun_Tokenizer 학습
noun_extractor = LRNounExtractor_v2(verbose=True)
nouns = noun_extractor.train_extract(contents, min_noun_frequency=50)
#
#
#
#
# Pickle 라이브러리 사용해서 저장
with open('./nouns_sk.data', 'wb') as f:
# with open('./nouns_posco.data', 'wb') as f:
    pickle.dump(nouns, f, pickle.HIGHEST_PROTOCOL)

#
# Pickle 라이브러리 사용해서 불러오기
# with open('./nouns_sk.data', 'rb') as f:
# with open('./nouns_posco.data', 'rb') as f:
#     nouns = pickle.load(f)
#
# # 사전 길이
# print(len(nouns))
#
# Tokenizer 호출해서 문장 분석
l_match_tokenizer = NounLMatchTokenizer(nouns)

for i in range(0, len(contents)):
    temp = l_match_tokenizer.tokenize(contents[i])
    print(temp)
    # temp2 = (' ').join(temp)+
    # row = []
    # f = open('./test.csv', 'a', encoding='UTF-8', newline='')
    # wr = csv.writer(f)
    # row.append(temp2)
    # row.append(points[i])
    # wr.writerow(row)
    # print(row)
#

# for j in text:
#     print(j, len(model.get_word_vector(j)))

# text = contents[0].split()

# print(text)

# print(len(model.get_labels())) # 사전 길이 체크
# print(model.get_dimension()) # 벡터차원 가져오기
#
# print(model.get_labels())

# for i in text:
#     print(i, model.get_word_vector(i))

# print(model.get_subwords('하락'))
