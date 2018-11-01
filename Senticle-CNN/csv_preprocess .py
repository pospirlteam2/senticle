# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:18:42 2018

@author: ydy89
"""
#%%

import pandas as pd
import os
import re
from bs4 import BeautifulSoup
import FinanceDataReader as fdr
import sklearn.preprocessing
import datetime
from konlpy.tag import Kkma, Okt


#####################################################
#                  데이터 정제                      #   
#####################################################

#데이터 불러오기(포스코.csv 파일 이름 바꿈)
df = pd.read_csv('POSCO_4y.csv')
df = df.iloc[:,:2]
company = '포스코'
df.columns = ['datetime','text']

df.datetime = df.datetime.apply(lambda x:str(x))


df.datetime = pd.to_datetime(df.datetime)

df = df.sort_values('datetime')
df = df.dropna()
#df = df.reset_index()
count = 0
while True:
    count+=1
    try:
        print("%d번째 바퀴 (거 쫌만 기다리소)"%count)
        for i in range(0, len(df)-1):
            if df.iloc[i].text[:50]==df.iloc[i+1].text[:50]:
                df = df.drop(df.iloc[i+1].name, axis = 0)
    except IndexError:
        pass
    else:
        break
 
#날짜 앞에 '20'붙이기
#df['datetime'] = df['datetime'].apply(lambda x: '20'+str(x))

#날짜컬럼 인덱스로 설정
df = df.set_index('datetime')
#인덱스 전부 Datetime 형식으로
df.index = pd.to_datetime(df.index)

#데이터프레임 인덱스 소팅(내림차순)
df = df.sort_index()

#학습데이터의 처음 3시 30 이전 데이터는 버리기 
if len(df[df.index<=datetime.datetime(year = df.index[0].year, month = df.index[0].month, day = df.index[0].day, hour = 15, minute = 30)].index)==0:
    pass
else:
    df = df.drop(df[df.index<=datetime.datetime(year = df.index[0].year, month = df.index[0].month, day = df.index[0].day, hour = 15, minute = 30)].index, axis = 0)
#df.info()
#df.head(10) 

#학습데이터의 마지막 3:30 이후 데이터는 버리기
if len(df[df.index>=datetime.datetime(year = df.index[-1].year, month = df.index[-1].month, day = df.index[-1].day, hour = 15, minute = 30)].index)==0:
    pass
else:
    df = df.drop(df[df.index>=datetime.datetime(year = df.index[-1].year, month = df.index[-1].month, day = df.index[-1].day, hour = 15, minute = 30)].index, axis = 0)
#%%
min_date = df.index.date[0]
max_date = df.index.date[-1]+datetime.timedelta(days=7) 
stock_df = fdr.DataReader('005940',min_date, max_date)

#Change가 0인 행 모두 삭제 (나중에 병합시 nan값으로 만들고, 채워주기 위함)
stock_df = stock_df.drop(stock_df[stock_df.Change==0].index, axis = 0)
stock_df.index = stock_df.index.date


df_copy = df.copy()
df_copy['impact_date'] = df_copy.index.date
for i in range(len(df_copy)):
    if datetime.time(df_copy.index[i].hour, df_copy.index[i].minute) >= datetime.time(hour = 15, minute = 30):
        df_copy['impact_date'][i] = df_copy['impact_date'][i] + datetime.timedelta(1)

def updown(x):
    if x>0:
        return 1
    else:
        return 0
df_copy = df_copy.set_index('impact_date')
stock_df['num'] = stock_df['Change'].apply(lambda x: updown(x))
df_copy['num'] = stock_df['num']
df_copy['num'] = df_copy['num'].fillna(method = 'bfill')
if df_copy.num.isnull().sum()>=1:
    nextdate = df_copy.index[-1]+datetime.timedelta(1)
    while nextdate not in stock_df.index: 
        nextdate = nextdate + datetime.timedelta(1)
    df_copy.num[-1] = stock_df.loc[nextdate].num
df_copy['num'] = df_copy['num'].fillna(method = 'bfill')
#
bin_df = pd.DataFrame({'text':[]})
for i in range(len(df_copy)):
    print("총 %d개 기사 중 %d번째 기사 특수문자 제거중....."%(len(df_copy),i))
    string_soup = BeautifulSoup(df_copy.iloc[i].text,'html5lib')
    newstring = re.sub('[^가-힣 A-Z\n]',' ', string_soup.get_text())
    bin_df =bin_df.append(pd.DataFrame({'text':newstring}, index = [df_copy.index[i]]))

bin_df['num'] = df_copy.num.apply(lambda x : int(x))
bin_df.index = df.index

#여까지는 완성 bin_df가 진퉁
#%% 유사도 점검을 통한 중복기사 제거
#from sklearn.feature_extraction.text import TfidfVectorizer
#kkma = Kkma()
#
#mydoclist_kkma = []
#for i in range(len(bin_df.text)):
#    print('전체 %d개 기사 중 %d번째 기사 형태소 분석 중...'%(len(bin_df.text),i))
#    kkma_nouns = ' '.join(kkma.nouns(bin_df.text[i]))
#    mydoclist_kkma.append(kkma_nouns)
#
#tfidf_vectorizer = TfidfVectorizer(min_df = 1)
#tfidf_matrix_kkma = tfidf_vectorizer.fit_transform(mydoclist_kkma)
#
#document_distances_kkma = (tfidf_matrix_kkma * tfidf_matrix_kkma.T)
#print('\n')
#print('kkma를 활용한 유사도 분석을 위해 '+str(document_distances_kkma.get_shape()[0])+'x'+str(document_distances_kkma.get_shape()[1])+'matrix를 만들었따뤼~ ^오^')
#print('\n')
#print(document_distances_kkma.toarray())
#ar = document_distances_kkma.toarray()
#ar_lst = []
#for i in range(len(bin_df)):
#    print('전체 %d개 인덱스 중 %d번째 인덱스 체크 중...'%(len(bin_df),i))
#    for j in range(i+1,len(ar)):
#        if ar[i][1+j-1]>0.24:
#            ar_lst.append(j)
#ar_lst = list(set(ar_lst))
#len(ar)-len(ar_lst)
#
#bin_df = bin_df.reset_index()
#bin_df = bin_df.drop(bin_df.index[ar_lst],axis = 0)
#bin_df = bin_df.set_index('datetime')
#%% text길이 5개 미만인 거 다 지우기기
bin_df = bin_df.reset_index()
try :
    for i in range(len(bin_df)):
        if len(bin_df.iloc[i].text)<5:
            bin_df = bin_df.drop(bin_df.index[i],axis = 0)
except IndexError:
    pass

bin_df = bin_df.set_index('datetime')
bin_df.to_csv(r'posco_4Y_data.csv', index = True, header=True)
#%
#%%
# 여기까지는 하루 단위
##%% 여기부터는 세시 반으로 묶어서(하루단위로) 처리하는 거
#
##데이터프레임 오후 3:30 기준으로 텍스트 묶기
#def date_categorize(df):
#    df_copy = df.copy()
#    df_copy['impact_date'] = df_copy.index.date
#    for i in range(len(df_copy)):
#        if datetime.time(df_copy.index[i].hour, df_copy.index[i].minute) >= datetime.time(hour = 15, minute = 30):
#            df_copy['impact_date'][i] = df_copy['impact_date'][i] + datetime.timedelta(1)
#    df_copy = df_copy.set_index('impact_date') #인덱스 재설정
#    df_copy = df_copy.groupby('impact_date').sum()
#    return df_copy
#
#newdf = date_categorize(df)
#newdf #날짜별 텍스트 다모은거 
#newdf

################################################################
##             뉴스 데이터 병합 및 주가 상/하락 병합           #
################################################################
#
##날짜 선정
#    #기간은 불러온 csv파일 기준 하루 밀린 날짜범위
#min_date = newdf.index[0]#+datetime.timedelta(days=1)
#
#    #포스코 20일처럼 거래가 있었지만 change가 0일때도 있음. 이러면 다음날로 한번 더넘겨야해서 +2
#max_date = newdf.index[-1]+datetime.timedelta(days=7) 
#
##주식 소환
#stock_df = fdr.DataReader('005490',min_date, max_date)
#
#
##Change가 0인 행 모두 삭제 (나중에 병합시 nan값으로 만들고, 채워주기 위함)
#stock_df = stock_df.drop(stock_df[stock_df.Change==0].index, axis = 0)
#
##인덱스 데이터형식 맞추기
#if stock_df.index[1] == newdf.index[1]:# 주가df랑 뉴스df랑 인덱스 형식 같으면 True나옴
#    pass
#else:
#    stock_df.index = stock_df.index.date #한번만 실행해야 오류 안남
#
#print('--'*10 +'주식 데이터 상위(20개)'+'--'*10)
#print(stock_df.head(20))
#
#
#########################################################
##가상 환경 (마지막 행이 nan으로 끝나는 상황 가정) 
##stock_df = stock_df.drop(stock_df.index[-2], axis = 0)
#########################################################
#
#

#기사 데이터프레임에 change 컬럼 생성(주식df 기준)
#newdf['Change'] = stock_df['Change'] #이때 NaN으로 나온다는 것 : 뉴스O, 주식시장 개장 X
#
##NaN으로 나온 change값을 다음날의 값으로 합치기
#newdf['Change'] = newdf['Change'].fillna(method = 'bfill')
#
##newdf의 마지막 change값이 nan으로 끝날 때(위의 fiina사용 못할때) 처리하기
#if newdf.Change.isnull().sum()>=1:
#    nextdate = newdf.index[-1]+datetime.timedelta(1)#20일
#    while nextdate not in stock_df.index: #nan값으로 뜬 다음날이 주가에 없으면 진입
#        nextdate = nextdate + datetime.timedelta(1)
#    newdf.Change[-1] = stock_df.loc[nextdate].Change
## 한번더 NaN 제거 : 맨 아래 NaN가 연속 두개로 끝나면, 위의 if문 실행 시 마지막꺼만 채워지기때문에
#    #NaN
#    #0.0011 이런식으로 끝나기 때문에 한번더 채워줘야함
#newdf['Change'] = newdf['Change'].fillna(method = 'bfill')
#
#print('--'*20 +'뉴스 기사 데이터(후)'+'--'*20)
#print(newdf) 

#
#####################################################################
##         뉴스데이터 상승/하락으로 분류 & 자연어처리               #
#####################################################################
#
#def make_dic(pos, dic):
#    for x in pos:
#        if x not in dic:
#            dic[x] = 1
#        else:
#            dic[x]+=1    
#    return dic
#
#def normalize_data(df):
#    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
#    #min_max_scaler = sklearn.preprocessing.normalize()
#    df['text'] = min_max_scaler.fit_transform(df['text'].values.reshape(-1,1))
#    return df
#def dic_to_df_per_new(df, pos_dic, pos_col):
#    for i in range(1,len(df)):
#        start_times = time.time()
#        string_soup = BeautifulSoup(df.text[i],'html5lib')
#        newstring = re.sub('[^가-힣 a-zA-Z\.\n]',' ', string_soup.get_text())
#        
#        pos_dic = {}
#        pos_string = kkma.pos(newstring)
#        pos_string = sorted(pos_string)
#        #딕셔너리 만들기
#        pos_dic = make_dic(pos_string, pos_dic)
#        pos_col = pos_col.append(pd.DataFrame([pos_dic]))
#        print("(%d번째/총 %d개 뉴스 중) POS 분석 소요시간 : %s 초" %((i),len(df),round((time.time() - start_times),3)))
#    
#    return pos_col
#
#
#up_news_df = pd.DataFrame({'text':''}, index = newdf[newdf.Change>0].index)
#down_news_df = pd.DataFrame({'text':''}, index = newdf[newdf.Change<0].index)
#
##상승 하락별로 데이터프레임 채우기
#up_news_df.text = newdf['text'] 
#down_news_df.text = newdf['text']
#
##upnews와 downnews 사이즈 조절
#length = min(len(up_news_df),len(down_news_df))
#up_news_df = up_news_df[-length:]
#down_news_df = down_news_df[-length:]
#
##아래 make_dic을 실행하기 위한 초기화 작업
#string_soup_up = BeautifulSoup(up_news_df.text[0],'html5lib')
#string_soup_down = BeautifulSoup(down_news_df.text[0],'html5lib')
#newstring_up = re.sub('[^가-힣 a-zA-Z\.\n]',' ', string_soup_up.get_text())
#newstring_down = re.sub('[^가-힣 a-zA-Z\.\n]',' ', string_soup_down.get_text())
#
#up_pos_dic = {}
#down_pos_dic = {}
#kkma = Kkma()
#pos_string_up = kkma.pos(newstring_up)
#pos_string_down = kkma.pos(newstring_down)
#
#pos_string_up = sorted(pos_string_up)
#pos_string_down = sorted(pos_string_down)
#
#
#
##딕셔너리 만들기
#pos_dic_up = make_dic(pos_string_up, up_pos_dic)
#pos_col_up = pd.DataFrame([pos_dic_up], index = [up_news_df.index[0]])
#
#pos_dic_down = make_dic(pos_string_down, down_pos_dic)
#pos_col_down = pd.DataFrame([pos_dic_down], index = [down_news_df.index[0]])
#
##인덱스 별 딕셔너리 추가 
#pos_col_up = dic_to_df_per_new(up_news_df,pos_dic_up, pos_col_up)
#pos_col_down = dic_to_df_per_new(down_news_df,pos_dic_down,pos_col_down)
#
#real_pos_up = pos_col_up.copy()
#real_pos_down = pos_col_down.copy()
#
##셋 인덱스 가져오기 
#real_pos_up = real_pos_up.set_index(up_news_df.index)
#real_pos_down = real_pos_down.set_index(down_news_df.index)
#
#used_word = ['NNG','NNP','NNM','NR','NP','VV','VA','VXV','VXA']
#ComName_pos = kkma.pos(company)
#stop_word_index_up = []
#stop_word_index_down = []
#for i in real_pos_up.columns:
#    if i[1] not in used_word:
#        stop_word_index_up.append(i)
#for i in ComName_pos:
#    stop_word_index_up.append(i)  
#    
#for i in real_pos_down.columns:
#    if i[1] not in used_word:
#        stop_word_index_down.append(i)
#for i in ComName_pos:
#    stop_word_index_down.append(i)   
#stop_word_index_up.append(('억','NR'))
#stop_word_index_up.append(('이','VCP'))
#stop_word_index_up.append(('있','VXV'))
#stop_word_index_up.append(('원','NNM'))
#stop_word_index_up.append(('이','NNG'))
#
#stop_word_index_down.append(('억','NR'))
#stop_word_index_down.append(('이','VCP'))
#stop_word_index_down.append(('있','VXV'))
#stop_word_index_down.append(('원','NNM'))
#stop_word_index_down.append(('이','NNG'))
#
#
#real_pos_up = real_pos_up.drop(real_pos_up[stop_word_index_up], axis = 1)
#real_pos_down = real_pos_down.drop(real_pos_down[stop_word_index_down], axis = 1)
#
### 형태소 갯수 합친 sum 컬럼 만들기
#real_pos_up['sum'] = real_pos_up.sum(axis=1)
#real_pos_down['sum'] = real_pos_down.sum(axis=1)
#
#real_pos_up1 = real_pos_up.copy()
#norm_pos_up = real_pos_up1.div(real_pos_up1.sum(axis = 1), axis = 0)
#
#real_pos_down1 = real_pos_down.copy()
#norm_pos_down = real_pos_down1.div(real_pos_down1.sum(axis = 1), axis = 0)
#
#
#up_pos_sum = norm_pos_up.sum(axis = 0)
#down_pos_sum = norm_pos_down.sum(axis=0)
#
#up_pos_sum[:-1].plot()
#down_pos_sum[:-1].plot() 
#
#up_pos_sum[up_pos_sum>0.3]
#
#up_pos_df = up_pos_sum.to_frame('UpPos')
#down_pos_df = down_pos_sum.to_frame('DownPos')
#down_pos_df = down_pos_df*(-1)
#
##NaN값 0으로 바꿔주기
#dictionary = pd.concat([up_pos_df, down_pos_df], axis = 1)
#dictionary = dictionary.fillna(0)
#dictionary['sum'] = dictionary.UpPos + dictionary.DownPos
#dictionary = dictionary.iloc[1:,:]['sum']
#
#dictionary
#
###########################################################################
##           RawData인 기사 데이터에 스코어 컬럼 추가하는 for문           #
###########################################################################
#print('*'*60)
#print(' '*10+"뉴스별 긍/부정 스코어 계산중...ZZZZzzzz..."+' '*10)
#print('*'*60)
#
#
#start_time = time.time()
#Score_col = pd.DataFrame({'Score':[]})
#
#
#for i in range(len(newdf)):
#    start_time_news = time.time()
#    string = newdf.text[i]
#    stringSoup = BeautifulSoup(string,'html5lib')
#    
#    #잡다한거 다 지운 스트링
#    newstring = re.sub('[^가-힣 a-zA-Z\.\n]',' ', stringSoup.get_text())
#    
#    newstring1 = kkma.pos(newstring)
#    
#    test_dic = {}
#    
#    string1 = newstring1.copy()
#    
#    string1 = sorted(string1)
#    
#    test_dic = make_dic(string1,test_dic) #신문기사 하나에서 만든 
#    
#    testDF = pd.Series(test_dic, index=test_dic.keys())#df로 만들기
#    testDF = testDF.to_frame(name='text')
#    
#    testDF['norm'] = dictionary#기사하나의 형태소 분석 + 사전컬럼 합지기
#    #testDF 출력해보면 NaN이 있는데 이건 dictionary는 불용어 제거하고, testDF는 안해서그럼
#    #그냥 dropna로 날려주면됨
#    
#    testDF['score'] = normalize_data(testDF).text * testDF.norm  
#    testDF_a = testDF['score']
#    testDF_a = testDF_a.dropna(how = 'any')
#    
#    score_of_news = testDF_a.sum() 
#    Score_col = Score_col.append(pd.DataFrame({'Score':score_of_news}, index = [newdf.index[i]]))
#    print("(%d번째/총 %d개 뉴스 중) 스코어링 소요시간 : %s 초" %((i+1),len(newdf),round((time.time() - start_time_news),3)))
#
#
#Score_col
#score_newdf = newdf.copy()
#score_newdf['Score'] = Score_col
#score_newdf = score_newdf[['Change','Score']]
#score_newdf.Change
#print("전체 스코어링 소요시간 : %s 초" %round((time.time() - start_time),3))
#%%

#*************************************************
#*************************************************
#*************************************************
#*************************************************
#                                                #
#                 TF-IDF 고려....                #
#                                                #
#*************************************************
#*************************************************
#*************************************************
#*************************************************
#*************************************************

##문제 발생 : 일단 상관계수가 씹창남;
#score_newdf.corr(method = 'pearson')
#score_newdf[['Change','Score']].plot()
##%%
################################################################################
################################################################################
################################################################################
################################################################################
##                                                                             #
##                             여기다 쓰애끼야                                 #
##                                                                             #
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
#
## In[1]:
#os.chdir('/home/pirl/Downloads/뉴감주예/nlp-tensorflow-master/01-sentiment_analysis/')
#import tensorflow as tf
#from data_process import build_vocab, batch_iter, sentence_to_index
#from models import LSTM, biLSTM, deepBiLSTM
#
#
## In[2]:
#
#train = pd.read_csv('./data/train-5T.txt', delimiter='\t')
#test = pd.read_csv('./data/test-1T.txt', delimiter='\t')
#train
##%%
#newdf.Change = score_newdf.Score
#newdf.columns = ['document','label']
#train = newdf[:-1]
#test = newdf[-1:]
#
## In[4]:
#
#
#X_train = train.document
#Y_train = train.label
#X_test = test.document
#Y_test = test.label
#
#
## In[5]:
#
#
#max_vocab = 50000
#vocab, _, vocab_size = build_vocab(X_train, max_vocab)
#
#
## # Sentiment Analysis with LSTM
#
## In[6]:
#
#
#batches = batch_iter(list(zip(X_train, Y_train)), batch_size=64, num_epochs=15)
#
#
## In[7]:
#
#
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.reset_default_graph()
#sess = tf.Session(config=config)
#model = LSTM(sess=sess, vocab_size=vocab_size, lr=1e-2)
#train_loss = []
#train_acc = []
#test_loss = []
#test_acc = []
#
#for step, batch in enumerate(batches):
#    x_train, y_train = zip(*batch)
#    x_train = sentence_to_index(x_train, vocab)
#    acc = model.get_accuracy(x_train, y_train)
#    l, _ = model.train(x_train, y_train)
#    train_loss.append(l)
#    train_acc.append(acc)
#    
#    if step % 100 == 0:
#        test_batches = batch_iter(list(zip(X_test, Y_test)), batch_size=64, num_epochs=1)
#        for test_batch in test_batches:
#            x_test, y_test = zip(*test_batch)
#            x_test = sentence_to_index(x_test, vocab)
#            t_acc = model.get_accuracy(x_test, y_test)
#            t_loss = model.get_loss(x_test, y_test)
#            test_loss.append(t_loss)
#            test_acc.append(t_acc)
#        print('batch:', '%04d' % step, '\ntrain loss:', '%.5f' % np.mean(train_loss), 
#              '\ttest loss:', '%.5f' % np.mean(test_loss))
#        print('train accuracy:', '%.3f' % np.mean(train_acc), '\ttest accuracy:', 
#              '%.3f' % np.mean(test_acc), '\n')
#        train_loss = []
#        train_acc = []
#        test_loss = []
#        test_acc = []
#
#
## # Sentiment Analysis with biLSTM
#
## In[ ]:
#
#
#batches = batch_iter(list(zip(X_train, Y_train)), batch_size=64, num_epochs=15)
#
#
## In[ ]:
#
#
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.reset_default_graph()
#sess = tf.Session(config=config)
#model = biLSTM(sess=sess, vocab_size=vocab_size, lr=1e-2)
#train_loss = []
#train_acc = []
#test_loss = []
#test_acc = []
#
#for step, batch in enumerate(batches):
#    x_train, y_train = zip(*batch)
#    x_train = sentence_to_index(x_train, vocab)
#    acc = model.get_accuracy(x_train, y_train)
#    l, _ = model.train(x_train, y_train)
#    train_loss.append(l)
#    train_acc.append(acc)
#    
#    if step % 100 == 0:
#        test_batches = batch_iter(list(zip(X_test, Y_test)), batch_size=64, num_epochs=1)
#        for test_batch in test_batches:
#            x_test, y_test = zip(*test_batch)
#            x_test = sentence_to_index(x_test, vocab)
#            t_acc = model.get_accuracy(x_test, y_test)
#            t_loss = model.get_loss(x_test, y_test)
#            test_loss.append(t_loss)
#            test_acc.append(t_acc)
#        print('batch:', '%04d' % step, '\ntrain loss:', '%.5f' % np.mean(train_loss), 
#              '\ttest loss:', '%.5f' % np.mean(test_loss))
#        print('train accuracy:', '%.3f' % np.mean(train_acc), '\ttest accuracy:', 
#              '%.3f' % np.mean(test_acc), '\n')
#        train_loss = []
#        train_acc = []
#        test_loss = []
#        test_acc = []
#
#
## # Sentiment Analysis with deepBiLSTM
#
## In[ ]:
#
#
#batches = batch_iter(list(zip(X_train, Y_train)), batch_size=64, num_epochs=15)
#
#
## In[ ]:
#
#
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.reset_default_graph()
#sess = tf.Session(config=config)
#model = deepBiLSTM(sess=sess, vocab_size=vocab_size, lr=1e-2)
#train_loss = []
#train_acc = []
#test_loss = []
#test_acc = []
#
#for step, batch in enumerate(batches):
#    x_train, y_train = zip(*batch)
#    x_train = sentence_to_index(x_train, vocab)
#    acc = model.get_accuracy(x_train, y_train)
#    l, _ = model.train(x_train, y_train)
#    train_loss.append(l)
#    train_acc.append(acc)
#    
#    if step % 100 == 0:
#        test_batches = batch_iter(list(zip(X_test, Y_test)), batch_size=64, num_epochs=1)
#        for test_batch in test_batches:
#            x_test, y_test = zip(*test_batch)
#            x_test = sentence_to_index(x_test, vocab)
#            t_acc = model.get_accuracy(x_test, y_test)
#            t_loss = model.get_loss(x_test, y_test)
#            test_loss.append(t_loss)
#            test_acc.append(t_acc)
#        print('batch:', '%04d' % step, '\ntrain loss:', '%.5f' % np.mean(train_loss), 
#              '\ttest loss:', '%.5f' % np.mean(test_loss))
#        print('train accuracy:', '%.3f' % np.mean(train_acc), '\ttest accuracy:', 
#              '%.3f' % np.mean(test_acc), '\n')
#        train_loss = []
#        train_acc = []
#        test_loss = []
#        test_acc = []
#
