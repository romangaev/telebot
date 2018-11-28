import pandas as pd
import gensim
from bson import Binary
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import *
import numpy as np
np.random.seed(2018)
from nltk.corpus import stopwords
from collections import defaultdict
import wikipedia
import pickle

from pymongo import MongoClient



NUMBER_OF_CLUSTERS = 35

# методы для предобработки текста
def lemmatize_stemming(text):
    stemmer = SnowballStemmer("russian")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(str(text)):
        if "http" not in token:
            if token not in stopwords.words("russian") and len(token) > 3:
                stemmed = lemmatize_stemming(token)
                full_word_dictionary[stemmed] = token
                result.append(stemmed)
    return result

if __name__ == '__main__':
    client = MongoClient('mongodb://rgaev:iha492081@ds141813.mlab.com:41813/digital_wallet')
    db = client.digital_wallet
    var_files_collection = db.var_files

    # читаем файл с данныим
    data = pd.read_csv('vk.csv', error_bad_lines=False)
    data_text = data[['question']]

    answers = data[['answer']]
    data_text['index'] = data_text.index
    answers['index'] = answers.index
    documents = data_text
    answers = data["answer"].tolist()
    #print(answers[:5])
    answers = [str(every)[:250] for every in answers]
    for i in range(10):
        print(answers[i])
        print('\n')

    # словарь всех кластеров с листами вопросов принадлежащих кластерам
    clusters_with_questions = defaultdict(list)

    # словарь для сохранения полной формы слова (stemmed - full form)
    full_word_dictionary = {}

    # токенизация и стемминг всех предложений
    processed_docs = documents['question'].map(preprocess)

    # составляем словарь слов с индексами (id - string)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # Bag of Words корпус и обработка в TFIDF
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    # делаем модель кластеризации по корпусу
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=NUMBER_OF_CLUSTERS, id2word=dictionary, passes=2, workers=4)


    # ОПРЕДЕЛЕНИЕ КЛАСТЕРА ДЛЯ КАЖДОГО ВОПРОСА В БАЗЕ
    for index, row in documents.iterrows():
            question_id= row["index"]
            question = row["question"]
            bow_vector = dictionary.doc2bow(preprocess(question))
            tfidf_vector = tfidf[bow_vector]
            predicted_topic_id = sorted(lda_model_tfidf.get_document_topics(bow_vector, minimum_probability=None, minimum_phi_value=None,
                                                             per_word_topics=False), key=lambda x: x[1], reverse=True)[0][0]
            clusters_with_questions[predicted_topic_id].append(question_id)

    # печать распределения вопросов по кластерам
    for key, value in clusters_with_questions.items():
        print('Номер кластера:')
        print(key)
        print('Количество вопросов относящихся к кластеру:')
        print(len(value))
        print('\n')

    print("Predicting unseen doc:")
    unseen_document = 'где находится ближайший офис сбербанка?'

    print(unseen_document)
    print(preprocess(unseen_document))
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    tfidf_vector = tfidf[bow_vector]
    print(bow_vector)


    print("Лист возможных топиков для предложения (get_document_topics):")
    answer_topics = sorted(
        lda_model_tfidf.get_document_topics(tfidf_vector, minimum_probability=None, minimum_phi_value=None,
                                            per_word_topics=False), key=lambda x: x[1], reverse=True)[:5]
    print(answer_topics)




    # занесение в базу
    post1 = {'name': "dictionary", 'file': Binary(pickle.dumps(dictionary))}
    var_files_collection.posts.insert_one(post1)

    post2 = {'name': "tfidf_model", 'file': Binary(pickle.dumps(tfidf))}
    var_files_collection.posts.insert_one(post2)

    post3 = {'name': "clusters_with_questions", 'file': Binary(pickle.dumps(clusters_with_questions))}
    var_files_collection.posts.insert_one(post3)

    post4 = {'name': "corpus_tfidf", 'file': Binary(pickle.dumps(corpus_tfidf))}
    var_files_collection.posts.insert_one(post4)

    post5 = {'name': "bow_corpus", 'file': Binary(pickle.dumps(bow_corpus))}
    var_files_collection.posts.insert_one(post5)

    post6 = {'name': "lda", 'file': Binary(pickle.dumps(lda_model_tfidf))}
    var_files_collection.posts.insert_one(post6)

    post7 = {'name': "documents", 'file': Binary(pickle.dumps(documents))}
    var_files_collection.posts.insert_one(post7)

    post8 = {'name': "answers", 'file': Binary(pickle.dumps(answers))}
    var_files_collection.posts.insert_one(post8)

    post9 = {'name': "full_form",'file': Binary(pickle.dumps(full_word_dictionary))}
    var_files_collection.posts.insert_one(post9)
