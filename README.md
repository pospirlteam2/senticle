# senticle 
## 파일 설명
> 데이터는 'bigkinds'에서 수집한 SK하이닉스 기사와 POSCO 기사 
> python 3 


## Senticle-CNN
### cnn_tool.py
> main.py에서 사용하는 자연어 전처리 관련 함수들 



### crawler.py
> 실시간으로 뉴스기사를 크롤링해 서버에 저장 



### final_preprecess.py
> 주가와 관련하여 뉴스기사 라벨링



### main.py
> TextCNN 모델



### test.py
> 뉴스를 넣어 상하락 분류 테스트 



### train.py
> 트레이닝
> Flag를 이용해 파라미터 지정 





## crawler
### bigkinds_crawler.py
> https://www.kinds.or.kr 에서 keyword를 포함한 기사 수집 크롤러




### naver_crawler.py
> 네이버 증권뉴스 기사 수집 크롤러

