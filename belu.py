import collections
import json
import math
import jieba
import requests
import pandas as pd


def bleu_en(pred_seq, label_seq, k):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label -n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred -n + 1):
            if label_subs[' '.join(pred_tokens[i: i +n])] > 0:
                num_matches += 1
                label_subs[" ".join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred -n + 1), math.pow(0.5, n))
    return score


def bleu_cn(pred_seq, label_seq, k):
    pred_tokens, label_tokens = jieba.lcut(pred_seq, cut_all=False), jieba.lcut(label_seq, cut_all=False)

    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label -n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred -n + 1):
            if label_subs[' '.join(pred_tokens[i: i +n])] > 0:
                num_matches += 1
                label_subs[" ".join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred -n + 1), math.pow(0.5, n))
    return score


# engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
# preds = ['i like you too']
# labels = ['i love you too']
# for pred, label in zip(preds, labels):
#     score_en = bleu_en(pred, label, 2)
#
#
# preds = ['我去过中国北京天安门广场']
# labels = ['18. 我去过中国北京天安门广场']
# for pred, label in zip(preds, labels):
#     score_cn = bleu_cn(pred, label, 2)

url = "http://10.4.0.141:8000/v1/chat/completions"
files = ["/home/chentao/files/aaa/files/zh_2_en_test.json"]


def request_api(url, request_body):
    response = requests.post(url, json=request_body)  # 使用GET请求示例

    if response.status_code == 200:  # 判断响应状态码为200表示成功
        resp = json.loads(response.content)

        return resp['choices'][0]['message']['content'].strip("\n").strip(" ")
    else:
        print("Error occurred while calling the API.")
        return None


a1 = 0
a2 = 0
a3 = 0
a4 = 0
a5 = 0
a6 = 0
a7 = 0
a8 = 0
a9 = 0
a10 = 0

for file in files:
    with open(file, encoding='utf-8', mode='r') as f:
        datas = json.load(f)

    new_datas = []
    for data in datas:
        request_body = {
              "model": "string",
              "messages": [
                {
                  "role": "user",
                  "content": data['instruction']
                }
              ],
              "tools": [],
              "do_sample": True,
              "temperature": 0.95,
              "top_p": 0.7,
              "n": 1,
              "max_tokens": 0,
              "stream": False
            }
        predict = request_api(url, request_body)
        score_en = bleu_en(predict, data['output'], 2)
        if score_en <= 0.1:
            a1 += 1
        elif score_en <= 0.2:
            a2 += 1
        elif score_en <= 0.3:
            a3 += 1
        elif score_en <= 0.4:
            a4 += 1
        elif score_en <= 0.5:
            a5 += 1
        elif score_en <= 0.6:
            a6 += 1
        elif score_en <= 0.7:
            a7 += 1
        elif score_en <= 0.8:
            a8 += 1
        elif score_en <= 0.9:
            a9 += 1
        elif score_en <= 1:
            a10 += 1

print(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10)
        # new_datas.append({
        #     'instruction': data['instruction'],
        #     'output': data['output'],
        #     'predict': predict
        # })
