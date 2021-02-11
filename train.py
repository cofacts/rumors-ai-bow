import jieba
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import codecs
import numpy as np
from keras_bert import Tokenizer

VOCABULARY_PATH = './vocab.txt'

SEQUENCE_LENGTH = 128

token_dict = {}
with codecs.open(VOCABULARY_PATH, 'r', 'utf8') as file_reader:
    for line in file_reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict)


def bert_vectorize_data(articles_text):
    data_X = []

    for text in articles_text:
        ids, _ = tokenizer.encode(text, max_len=SEQUENCE_LENGTH)
        data_X.append(ids)

    data_X = np.array([data_X, np.zeros_like(data_X)])


SEQ_LEN = 1024

staging_articles = []
staging_x = []


article_list_files = sorted(os.listdir('../../../data/article_list'))

latest_article_list_filename = None if len(
    article_list_files) == 0 else article_list_files[-1]

last_updated = latest_article_list_filename[:-4]

article_list = pickle.load(
    open('../../../data/article_list/{}'.format(latest_article_list_filename), 'rb'))

for article_id in article_list:
    text = pickle.load(
        open('../../../data/articles/{}.pkl'.format(article_id), 'rb'))['text']
    ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
    staging_x.append(ids)


repos = ['raw_data']

repo_files = []
for repo in repos:
    files = os.listdir(repo)
    files = [file for file in files if 'json' in file]

    repo_files.append(files)

    print('There are ' + str(len(files)) + ' files in '+repo)

# load tags and text information of the files
# encoding: utf-8

repo = repos[0]
files = repo_files[0]

#df = pd.DataFrame()
#
# for file in files:
#    data = pd.read_json(os.path.join(repo, file))
#    data['file_name'] = file
#    df = df.append(data, ignore_index = True)

tag_type = 17
define_columns = ['id', 'text', 'tag_0', 'tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6',
                  'tag_7', 'tag_8', 'tag_9', 'tag_10', 'tag_11', 'tag_12', 'tag_13', 'tag_14', 'tag_15', 'tag_16']

data_list = []
for file in files:

    with open(os.path.join(repo, file), 'r', encoding='utf8') as f:
        data = json.load(f)

    tags_list = [0]*tag_type

    for tag in data['tags']:
        tags_list[tag] = 1

    # TBD: using file name or original id for modeling id
    #data_list.append([data['id'], data['text']] + tags_list)
    data_list.append([file, data['text']] + tags_list)

df_data = pd.DataFrame(data_list, columns=define_columns)

jieba.set_dictionary('./dict.txt.big.txt')

SEQ_LEN = 128


def load_data(df_dataset):

    tokenized_text = []

    indices, labels = [], []
    vectorizer = TfidfVectorizer()

    for row in df_dataset.iterrows():
        text = row[1]['text']
        tokenized_text.append(' '.join(jieba.cut(text, cut_all=True)))

        label = list(row[1].iloc[2:])
        label = label.index(max(label))
        labels.append(label)

    items = list(zip(tokenized_text, labels))

    np.random.shuffle(items)
    test_items = items[int(0.8*len(items)):]
    train_items = items[:int(0.8*len(items))]

    text_test, labels_test = zip(*test_items)
    text_train, labels_train = zip(*train_items)

    return vectorizer.fit_transform(text_train), labels_train, vectorizer.transform(text_test), labels_test


train_x, train_y, test_x, test_y = load_data(df_data)

clf = RandomForestClassifier(
    n_estimators=100, max_depth=25, min_samples_leaf=30, max_features=0.1)
clf.fit(train_x, train_y)
print(clf.score(train_x, train_y))
print(clf.score(test_x, test_y))

pickle.dump(vectorizer, open('model/bow/vectorizer.pkl', 'wb'))
pickle.dump(clf, open('model/bow/classifier.pkl', 'wb'))
