import random

import gensim
import pickle

from pymongo import MongoClient

client = MongoClient('mongodb://rgaev:iha492081@ds141813.mlab.com:41813/digital_wallet')
db = client.digital_wallet
var_files_collection = db.var_files

dictionary = pickle.loads(var_files_collection.posts.find_one({'name': "dictionary"})["file"])
clusters_with_questions = pickle.loads(
    var_files_collection.posts.find_one({'name': "clusters_with_questions"})["file"])
lda = pickle.loads(
    var_files_collection.posts.find_one({'name': "lda"})["file"])
tfidf_model = pickle.loads(
    var_files_collection.posts.find_one({'name': "tfidf_model"})["file"])
corpus_tfidf = pickle.loads(
    var_files_collection.posts.find_one({'name': "corpus_tfidf"})["file"])
documents = pickle.loads(
    var_files_collection.posts.find_one({'name': "documents"})["file"])
full_form = pickle.loads(
    var_files_collection.posts.find_one({'name': "full_form"})["file"])

for key, value in clusters_with_questions.items():
    print('Номер кластера:')
    print(key)
    print('Количество вопросов относящихся к кластеру:')
    print(len(value))


for j in range(1000):

        i=random.randint(0, 40000)

        # пробегаемся по каждому предложению в базе и ищем для него пару с вероятностью 1 в ближайших кластерах
        tfidf_vector=corpus_tfidf[i]
        answer_topics = sorted(
            lda.get_document_topics(tfidf_vector, minimum_probability=None, minimum_phi_value=None,
                                                per_word_topics=False), key=lambda x: x[1], reverse=True)[:5]

        answers_rating = []
        for every in answer_topics:
            topic_id = every[0]
            for question_id in clusters_with_questions[topic_id]:
                similarity = gensim.matutils.cossim(corpus_tfidf[question_id], tfidf_vector)
                if len(answers_rating) < 5:
                    answers_rating.append((question_id, similarity))
                    if len(answers_rating) == 5:
                        answers_rating.sort(key=lambda tup: tup[1], reverse=True)
                else:
                    if answers_rating[-1][1] < similarity:
                        answers_rating[-1] = (question_id, similarity)
                        answers_rating.sort(key=lambda tup: tup[1], reverse=True)

        res = False
        for every in answers_rating:
            if abs(every[1]-1.0)<0.1:
                res = True

        if res is False:
            print("False on "+str(i))
            break
