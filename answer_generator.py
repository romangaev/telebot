import pickle
import os
import gensim
import pandas as pd
from gensim.models import LdaModel, TfidfModel
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from pymongo import MongoClient


class AnswerGenerator:

    def __init__(self):
        client = MongoClient('mongodb://rgaev:iha492081@ds141813.mlab.com:41813/digital_wallet')
        db = client.digital_wallet
        var_files_collection = db.var_files

        self.dictionary = pickle.loads(var_files_collection.posts.find_one({'name': "dictionary"})["file"])
        #self.dictionary = pickle.load(open("dictionary.pickle", "rb"))
        self.clusters_with_questions = pickle.loads(var_files_collection.posts.find_one({'name': "clusters_with_questions"})["file"])
        #self.clusters_with_questions = pickle.load(open("clusters_with_questions.pickle", "rb"))
        self.lda = pickle.loads(
            var_files_collection.posts.find_one({'name': "lda"})["file"])
        #self.lda = LdaModel.load("my_model")
        self.tfidf_model = pickle.loads(
            var_files_collection.posts.find_one({'name': "tfidf_model"})["file"])
        #self.tfidf_model = TfidfModel.load("tfidf_model")
        self.corpus_tfidf = pickle.loads(
            var_files_collection.posts.find_one({'name': "corpus_tfidf"})["file"])
        #self.corpus_tfidf = pickle.load(open("corpus_tfidf.pickle", "rb"))

        self.documents = pickle.loads(
            var_files_collection.posts.find_one({'name': "documents"})["file"])

    def generate_answer(self, text):
        to_send = ''

        # векторизация текста
        bow_vector = self.dictionary.doc2bow(preprocess(text))
        tfidf_vector = self.tfidf_model[bow_vector]

        # Лист возможных топиков для предложения (get_document_topics)
        answer_topics = sorted(
            self.lda.get_document_topics(tfidf_vector, minimum_probability=None, minimum_phi_value=None,
                                                per_word_topics=False), key=lambda x: x[1], reverse=True)[:5]
        to_send = to_send+str(answer_topics)
        # Лист возможных вопросов для предложения:
        answers_rating = []
        for every in answer_topics:
            topic_id = every[0]
            for question_id in self.clusters_with_questions[topic_id]:

                similarity = gensim.matutils.cossim(self.corpus_tfidf[question_id], tfidf_vector)
                if len(answers_rating) < 5:
                    answers_rating.append((question_id, similarity))
                    if len(answers_rating) == 5:
                        answers_rating.sort(key=lambda tup: tup[1], reverse=True)
                else:
                    if answers_rating[-1][1] < similarity:
                        answers_rating[-1] = (question_id, similarity)
                        answers_rating.sort(key=lambda tup: tup[1], reverse=True)

        for every in answers_rating:
            question = self.documents['question'][every[0]]
            to_send = to_send + "\n"+"Вероятность: "+str(every[1])+"\n"+question

        return to_send

def lemmatize_stemming(text):
    stemmer = SnowballStemmer("russian")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(str(text)):
        if "http" not in token:
            if token not in stopwords.words("russian") and len(token) > 3:
                stemmed = lemmatize_stemming(token)
                result.append(stemmed)
    return result