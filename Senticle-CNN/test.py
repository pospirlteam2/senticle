import tensorflow as tf
import numpy as np
import cnn_tool as tool
from main import TextCNN
from lime.lime_text import LimeTextExplainer
import crawler as c
import requests
import json

SEQUENCE_LENGTH = 1400
NUM_CLASS = 2


def test(com):
    news_url = 'https://finance.naver.com'

    article = c.get_list(com)

    base_server = '182.215.14.185'
    URL = 'http://' + base_server + '/pos/index.php'

    with tf.Session() as sess:
        if com == 'posco':
            # 포스코 모델
            vocab = tool.load_vocab('news_vocab_posco.txt')
        else:
            # SK 모델
            vocab = tool.load_vocab('news_vocab_sk.txt')

        CNN = TextCNN(SEQUENCE_LENGTH, NUM_CLASS, len(vocab), 128, [3,4,5], 128)
        saver = tf.train.Saver()
        if com == 'posco':
            # 포스코 모델
            saver.restore(sess, './runs/1540105049/checkpoints/model-21800')
        else:
            # SK 모델
            saver.restore(sess, './runs/1540285428/checkpoints/model-2100')

        print('model restored')

        # article = c.get_list(com)

        for i in range(0, len(article)):
            input_text = article[i][4]

        # input_text = input('평가할 뉴스 입력 : ')

            tokens = tool.model_tokenize(input_text)

            print(tokens)

            sequence = [tool.get_token_id(t, vocab) for t in tokens]

            x = []
            while len(sequence) > 0:
                seq_seg = sequence[:SEQUENCE_LENGTH]
                sequence = sequence[SEQUENCE_LENGTH:]

                padding = [1] * (SEQUENCE_LENGTH - len(seq_seg))
                seq_seg = seq_seg + padding

                x.append(seq_seg)

            feed_dict = {
                CNN.input_x: x,
                CNN.dropout_keep_prob:1.0
            }

            predict = sess.run([CNN.predictions], feed_dict)

            result = np.mean(predict)

            if result == 1.0:
                print('하락')
            else:
                print('상승')

            test = sess.run(CNN.final, feed_dict)

            print(test)

            def predict_fn(x):
                predStorage = []
                for i in x:
                    tokens = tool.model_tokenize(i)
                    sequence = [tool.get_token_id(t, vocab) for t in tokens]
                    text = []
                    if len(sequence) > 0:
                        seq_seg = sequence[:SEQUENCE_LENGTH]
                        sequence = sequence[SEQUENCE_LENGTH:]

                        padding = [1] * (SEQUENCE_LENGTH - len(seq_seg))
                        seq_seg = seq_seg + padding

                        text.append(seq_seg)
                    else:
                        padding = [0] * (SEQUENCE_LENGTH)
                        text.append(padding)

                    feed_dict = {
                        CNN.input_x: text,
                        CNN.dropout_keep_prob: 1.0
                    }

                    scores = sess.run(CNN.final, feed_dict)


                    predStorage.append(np.squeeze(scores))

                return np.array(predStorage)

            explainer = LimeTextExplainer(class_names=['상승', '하락'])

            exp = explainer.explain_instance(input_text, predict_fn, num_features=6, num_samples=1400)
            key_list = exp.as_list()
            article[i].append(exp.as_list())
            article[i].append(np.squeeze(exp.predict_proba))

            if article[i][3] == '005490':
                company_name = 'posco'
            else:
                company_name = 'sk hynix'


            data2 = {'datetime': article[i][0], 'title': article[i][1], 'links': news_url + article[i][2], 'scores': '0.1',
                     'contents': article[i][4], 'company_name': company_name, 'up_prob': round(exp.predict_proba[0],2), 'down_prob': round(exp.predict_proba[1],2), 'word1': key_list[0][0],
                     'prob1': round(key_list[0][1],3), 'word2': key_list[1][0], 'prob2': round(key_list[1][1],3), 'word3': key_list[2][0], 'prob3': round(key_list[2][1],3), 'word4': key_list[3][0],
                     'prob4': round(key_list[3][1],3), 'word5': key_list[4][0], 'prob5': round(key_list[4][1],3), 'word6': key_list[5][0], 'prob6': round(key_list[5][1],3)}

            res = requests.post(URL, data=data2)
            print(res.content)
            # exp.save_to_file('./tmp/oi.html')

        # return data2

if __name__=='__main__':
    # temp = test()
    #
    # for i in temp:
    #     print(i)
    #
    #     print(i[5][0][0])
    #     print(i[6])
    #     # 0번째 : 날짜
    #     # 1번째 : 헤드라인
    #     # 2번쨰 : 링크
    #     # 3번째 : 종목코드 => posco
    #     # 4번째 : 기사내용
    #     # 5번째 : 키워드(리스트 분리 시급)
    #     # 6번째 : 점수(리스트)
    #
    #     # base_server = '182.215.14.185'
    #     # URL = 'http://' + base_server + '/pos/index.php'
    #     # data = {'company_name':'posco'}
    #     # data2 = {'datetime': i[0], 'title': i[1], 'links': i[2], 'scores': '0.1',
    #     #          'contents': i[4], 'company_name': 'posco', 'up_prob': i[6][0], 'down_prob': i[6][1], 'word1': i[5][0][0],
    #     #          'prob1': i[5][0][1], 'word2': i[5][1][0], 'prob2': i[5][1][1], 'word3': i[5][2][0], 'prob3': i[5][2][1], 'word4': i[5][3][0],
    #     #          'prob4': i[5][3][1], 'word5': i[5][4][0], 'prob5': i[5][4][1], 'word6': i[5][5][0], 'prob6': i[5][5][1]}
    #     # res = requests.post(URL, data2)
    #     # print(res.status_code)
    #     # print(res.content)
    #     # print(data2)
    # base_server = '182.215.14.185'
    # URL = 'http://' + base_server + '/pos/index.php'
    # res = requests.post(URL, data)
    # print(res.status_code)
    # print(data)
    company = input('회사명 입력(sk or posco) : ')
    data = test(company)