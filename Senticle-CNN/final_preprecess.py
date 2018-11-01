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
company = input("기업명 입력 영어로(ex.SKhynix) :")
nyun = input("몇 년치 출력?(최대 4년) : ")
week_val = input("주말을 뺀 기사만 고려할까요? (y/n) : ")
df = pd.read_csv(company+'_4y.csv')
df = df.iloc[:,:2]

df.columns = ['datetime','text']

df.datetime = df.datetime.apply(lambda x:str(x))


df.datetime = pd.to_datetime(df.datetime)

df = df.sort_values('datetime')
df = df.dropna()
count = 0
while True:
    count+=1
    try:
        print("%d번째 바퀴 (n번째랑 n+1번째랑 완전히 같은 기사만 삭제중...)"%count)
        for i in range(0, len(df)-1):
            if df.iloc[i].text[:50]==df.iloc[i+1].text[:50]:
                df = df.drop(df.iloc[i+1].name, axis = 0)
    except IndexError:
        pass
    else:
        break

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


#학습데이터의 마지막 3:30 이후 데이터는 버리기
if len(df[df.index>=datetime.datetime(year = df.index[-1].year, month = df.index[-1].month, day = df.index[-1].day, hour = 15, minute = 30)].index)==0:
    pass
else:
    df = df.drop(df[df.index>=datetime.datetime(year = df.index[-1].year, month = df.index[-1].month, day = df.index[-1].day, hour = 15, minute = 30)].index, axis = 0)

symbol = input("%s의 종목코드를 입력하세요 (6자리): "%company)
min_date = df.index.date[0]
max_date = df.index.date[-1]+datetime.timedelta(days=7) 
stock_df = fdr.DataReader(symbol,min_date, max_date)

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


bin_df = bin_df.reset_index()
try :
    for i in range(len(bin_df)):
        if len(bin_df.iloc[i].text)<5:
            bin_df = bin_df.drop(bin_df.index[i],axis = 0)
except IndexError:
    pass

bin_df = bin_df.set_index('datetime')
date = bin_df.index.date[-1]-datetime.timedelta(days=365*int(nyun))
year = date.year
mon = date.month
day = date.day
bin_df = bin_df[bin_df.index>=datetime.datetime(year, mon, day)]

if week_val.lower()=="y":    
    indx = []
    for i in bin_df.index:
        if i.weekday()==4 or i.weekday()==5:
            indx.append(i)
    for i in indx:
        try:
            bin_df = bin_df.drop(i,axis = 0)
        except KeyError:
            pass
else:
    pass
bin_df.to_csv(company+'_labeled_data.csv', index = True, header=True)
