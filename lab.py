import re
import jieba
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import csv


def read_json(path):
    f = open(path, 'rb')
    return json.load(f)

def analyzes(text):
    words = []
    labels = []

    for line in text:
        t = re.sub('[’@!,.:;\'\"#！？。：；、]+', '', line['content'])
        words.append(' '.join(jieba.lcut(t, cut_all=True)))

        if len(line) == 2:
            continue

        if line['label'] == 'positive':
            labels.append(1)
        elif line['label'] == 'neutral':
            labels.append(0)
        elif line['label'] == 'negative':
            labels.append(2)

    if len(line) == 2:
        return words
    else:
        return words, labels


def train_model(path):
    text = read_json(path)
    words, labels = analyzes(text)

    tf = TfidfVectorizer()
    x_train = tf.fit_transform(words).toarray()
    y_train = labels
    mt = MultinomialNB(alpha=0.1)
    mt.fit(x_train, y_train)
    return tf, mt


if __name__ == '__main__':
    tf, mt = train_model('train_data.json')

    test_text = read_json('test.json')
    test_words = analyzes(test_text)

    x_test = tf.transform(test_words).toarray()
    predict = mt.predict(x_test)

    ans = []
    for i in range(len(predict)):
        ans.append([i + 1, int(predict[i])])

    f = open('1180300315-周牧云.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerows(ans)
    f.close()
    print("finish")
