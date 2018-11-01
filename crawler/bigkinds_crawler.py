from bs4 import BeautifulSoup
import requests
import json
import re
import csv
from multiprocessing import Process

def get_url(c, b, s, e, p):
    if len(b) != 0:
        keyword = c
        ban = (' OR ').join(b)
        keyword += ' NOT ( ' + ban + ' )'
    else:
        keyword = c

    # 주소에 POST 방식으로 파라미터 전송
    url = 'https://www.kinds.or.kr/news/detailSearch.do'
    keyword_json = {
        "searchDetailTxt1": keyword,
        "agreeDetailTxt1": "",
        "needDetailTxt1": "",
        "exceptDetailTxt1": "",
        "o_id": "option1",
        "startDate": s,
        "endDate": e,
        "providerNm": "매일경제,머니투데이,서울경제,아시아경제,파이낸셜뉴스,한국경제,헤럴드경제",
        "categoryNm": "경제,자원,부동산,금융_재테크,경제일반,자동차,반도체,산업_기업,무역,서비스_쇼핑,증권_증시,외환,취업_창업,유통,국제경제",
        "incidentCategoryNm": "",
        "providerCode": "02100101,02100201,02100311,02100801,02100501,02100601,02100701",
        "categoryCode": "002000000,002004000,002010000,002008000,002014000,002011000,002009000,002005000,002001000,002012000,002006000,002002000,002007000,002003000,002013000",
        "incidentCategoryCode": "",
        "searchFtr": "1",
        "searchScope": "1",
        "searchKeyword": keyword
    }

    params = {
        'pageInfo': 'bksMain',
        'login_chk': 'null',
        'LOGIN_SN': 'null',
        'LOGIN_NAME': 'null',
        'indexName': 'news',
        'keyword': keyword, # 포스코 NOT (ICT OR 건설 OR 엠텍 OR 켐텍)
        'byLine':'',
        'searchScope': '1',
        'searchFtr': '1',
        'startDate': s,
        'endDate': e,
        'sortMethod': 'date',
        'contentLength': '100',
        'providerCode':'02100101,02100201,02100311,02100801,02100501,02100601,02100701',
        'categoryCode':'002000000,002004000,002010000,002008000,002014000,002011000,002009000,002005000,002001000,002012000,002006000,002002000,002007000,002003000,002013000',
        'incidentCode':'',
        'dateCode':'',
        'highlighting': 'true',
        'sessionUSID':'',
        'sessionUUID': 'test',
        'listMode':'',
        'categoryTab':'',
        'newsId':'',
        'delnewsId':'',
        'delquotationId':'',
        'delquotationtxt':'',
        'filterProviderCode':'',
        'filterCategoryCode':'',
        'filterIncidentCode':'',
        'filterDateCode':'',
        'filterAnalysisCode':'',
        'startNo': p,
        'resultNumber': '10',
        'topmenuoff':'',
        'resultState':'',
        # keywordJson: {"searchDetailTxt1":"포스코 NOT (ICT OR 건설 OR 엠텍 OR 켐텍)","agreeDetailTxt1":"","needDetailTxt1":"","exceptDetailTxt1":"","o_id":"option1","startDate":"2018-07-08","endDate":"2018-10-08","providerNm":"","categoryNm":"경제,자원,부동산,금융_재테크,경제일반,자동차,반도체,산업_기업,무역,서비스_쇼핑,증권_증시,외환,취업_창업,유통,국제경제","incidentCategoryNm":"","providerCode":"","categoryCode":"002000000,002004000,002010000,002008000,002014000,002011000,002009000,002005000,002001000,002012000,002006000,002002000,002007000,002003000,002013000","incidentCategoryCode":"","searchFtr":"1","searchScope":"1","searchKeyword":"포스코 NOT (ICT OR 건설 OR 엠텍 OR 켐텍)"}
        'keywordJson': keyword_json,
        'keywordFilterJson':'',
        'realKeyword':'',
        'keywordYn': 'Y',
        'totalCount':'',
        'interval':'',
        'quotationKeyword1':'',
        'quotationKeyword2':'',
        'quotationKeyword3':'',
        'printingPage':'',
        'searchFromUseYN': 'N',
        'searchFormName':'',
        'searchFormSaveSn':'',
        'mainTodayPersonYn':'',
        'period': '',
        'sectionDiv':''
    }

    response = requests.post(url=url, data=params)

    html = response.text

    soup = BeautifulSoup(html, 'html.parser')

    h3_list2 = soup.find_all('h3', attrs={'class': 'list_newsId'})

    url_list = []

    for id in h3_list2:
        url_list.append(id['id'].replace("news_", "docId="))

    return url_list

def get_article(l, f):
    base_url = 'https://www.kinds.or.kr/news/detailView.do?'

    article_text = []

    if l != None:
        article_url = base_url + l

        result = requests.get(article_url)

        data = json.loads(result.text)

        date = data['detail']['NEWS_ID']

        date2 = date.split('.')[1][0:14] #기사 작성 날짜

        text = data['detail']['CONTENT']

        text = re.sub('<.+?>', '', text, 0).strip() # 기사 내용

        text = re.sub('\[.*?]', '', text)

        switch = True

        while (switch):
            try:
                a = text.index('[')

                text = text[0:a]
            except(ValueError):
                switch = False

        text2 = text.split('.')

        for s in range(len(text2)-1, 0, -1):
            if '@' in text2[s]:
                for j in range(len(text2) - 1, s - 1, -1):
                    text2.pop(j)

        text3 = ('.').join(text2)

        article_text.append(str(date2))
        article_text.append(text3)

        print(article_text)

        f = open(f + '.csv', 'a', encoding='UTF-8', newline='')
        wr = csv.writer(f)
        wr.writerow(article_text)
    else:
        return 0


if __name__ == '__main__':
    company = input("회사명 입력 : ")
    # print(company)

    ban_list = []

    switch = True
    switch2 = True

    while(switch):
        keyword = input("금칙어 입력(입력안하면 스킵) : ")

        if keyword == '':
            switch = False
        else:
            ban_list.append(keyword)

    start = input('검색 시작일(0000-00-00) : ')

    end = input('검색 종료일(0000-00-00) : ')

    filename = input('저장할 파일 이름 : ')

    i = 1

    while(switch2):
        test2 = get_url(company, ban_list, start, end, i)

        if len(test2) != 0:
            procs = []

            for index, number in enumerate(test2):
                proc = Process(target=get_article, args=(number, filename))
                procs.append(proc)
                proc.start()

            for proc in procs:
                proc.join()
        else:
            switch2 = False

        print("%d 번째 페이지 크롤링 완료 !" %i)

        i += 1